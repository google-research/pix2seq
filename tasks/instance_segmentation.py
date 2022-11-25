# coding=utf-8
# Copyright 2022 The Pix2Seq Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Instance segmentation task via COCO metric evaluation."""

import copy
import os
import pickle

from absl import logging
import ml_collections
import numpy as np
import utils
import vocab
from data import data_utils
from metrics import metric_registry
from tasks import task as task_lib
from tasks import task_utils
from tasks.visualization import vis_utils
from pycocotools import _mask
from pycocotools import mask as mask_lib
import tensorflow as tf


@task_lib.TaskRegistry.register('instance_segmentation')
class TaskInstanceSegmentation(task_lib.Task):
  """Instance segmentation with coco metric evaluation."""

  def __init__(self,
               config: ml_collections.ConfigDict):
    super().__init__(config)
    if config.task.get('max_seq_len', 'auto') == 'auto':
      self.config.task.max_seq_len = 5 + config.task.max_points_per_object * 2
    self._category_names = task_utils.get_category_names(
        config.dataset.get('category_names_path'))
    metric_config = config.task.get('metric')
    if metric_config and metric_config.get('name'):
      self._coco_metrics = metric_registry.MetricRegistry.lookup(
          metric_config.name)(config)
    else:
      self._coco_metrics = None

  def preprocess_single(self, dataset, batch_duplicates, training):
    """Task-specific preprocessing of individual example in the dataset.

    Args:
      dataset: A tf.data.Dataset.
      batch_duplicates: `int`, enlarge a batch by augmenting it multiple times
        (as specified) and concating the augmented examples.
      training: bool.

    Returns:
      A dataset.
    """
    # Keep track of the base image size before any potential cropping is
    # performed. This is used for computing the mask size for metrics.
    def add_image_size_before_cropping(features, labels):
      features['image_size_before_cropping'] = tf.shape(features['image'])[:2]
      return features, labels
    dataset = dataset.map(add_image_size_before_cropping)

    dataset = data_utils.maybe_unbatch_instances_and_crop_image_to_bbox(
        dataset, self.config)

    def _preprocess_single_example(features, labels):
      config = self.config.task
      features['orig_image_size'] = tf.shape(features['image'])[:2]
      max_instances_per_image = (
          config.max_instances_per_image if training else
          config.max_instances_per_image_test)

      # Randomly shuffle the instances.
      logging.info('Shuffling instances')
      num_instances = tf.shape(labels['label'])[0]
      features, labels = copy.copy(features), copy.copy(labels)
      indices = tf.random.shuffle(tf.range(num_instances, dtype=tf.int32))
      for key in labels:
        labels[key] = tf.gather(labels[key], indices)

      # Sort instances by score.
      # Note: When using groundtruth boxes, scores == 1.0 in which case this
      # is a no-op. When using precomputed boxes, sorting by scores is helpful
      # for obtaining high quality instances.
      logging.info('Sorting labels by scores')
      indices = tf.argsort(labels['scores'], direction='DESCENDING')
      indices = indices[:max_instances_per_image]
      for key in labels:
        labels[key] = tf.gather(labels[key], indices)
      labels = data_utils.truncate_or_pad_to_max_instances(
          labels, max_instances_per_image)

      points_orig = labels['polygon']
      if training:
        features_list, labels_list = [], []
        for _ in range(batch_duplicates):
          features_, labels_ = data_utils.preprocess_train(
              features,
              labels,
              max_image_size=config.image_size,
              max_instances_per_image=max_instances_per_image,
              object_order=None,  # No reordering as `preserve_reserved_tokens`
              jitter_scale=(config.jitter_scale_min, config.jitter_scale_max),
              random_flip=True,
              color_jitter_strength=config.color_jitter_strength,
              filter_invalid_labels=True,
              object_coordinate_keys=('bbox', 'polygon', 'keypoints'))
          features_list.append(features_)
          labels_list.append(labels_)
        features = utils.merge_list_of_dict(features_list)
        labels = utils.merge_list_of_dict(labels_list)
      else:
        features, labels = data_utils.preprocess_eval(
            features,
            labels,
            max_image_size=config.image_size,
            max_instances_per_image=max_instances_per_image,
            object_coordinate_keys=('bbox', 'polygon', 'keypoints'))
      labels['polygon'] = utils.preserve_reserved_tokens(
          labels['polygon'], points_orig)

      return features, labels

    if training:
      dataset = dataset.filter(  # Filter out images with no annotations.
          lambda features, labels: tf.shape(labels['label'])[0] > 0)
    dataset = dataset.map(_preprocess_single_example,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def preprocess_batched(self, batched_examples, training):
    """Task-specific preprocessing of batched examples on accelerators (TPUs).

    Args:
      batched_examples: tuples of feature and label tensors that are
        preprocessed, batched, and stored with `dict`.
      training: bool.

    Returns:
      images: `float` of shape (bsz, h, w, c)
      input_seq: `int` of shape (bsz, instances, seqlen).
      target_seq: `int` of shape (bsz, instances, seqlen).
      token_weights: `float` of shape (bsz, instances, seqlen).
    """
    config = self.config.task
    mconfig = self.config.model
    features, labels = batched_examples
    prompt_seq = task_utils.build_instance_prompt_seq(
        self.task_vocab_id, labels['bbox'], labels['label'],
        config.quantization_bins, mconfig.coord_vocab_shift)
    original_points = labels['polygon']
    assert original_points.dtype == tf.float32
    response_seq = task_utils.build_instance_response_seq_from_points(
        original_points, labels['label'],
        config.quantization_bins, mconfig.coord_vocab_shift)
    label_seq = tf.concat([prompt_seq, response_seq], -1)

    # Pad sequence to a unified maximum length.
    assert label_seq.shape[-1] <= config.max_seq_len + 1
    label_seq = utils.pad_to_max_len(label_seq, config.max_seq_len + 1, -1)
    input_seq, target_seq = label_seq[..., :-1], label_seq[..., 1:]

    token_weights = tf.concat([tf.zeros_like(prompt_seq)[..., :-1],
                               tf.ones_like(response_seq)], -1)
    token_weights = tf.cast(token_weights, tf.float32)
    pad_t = tf.expand_dims(tf.greater(labels['label'], 0), -1)
    token_weights = tf.where(pad_t, token_weights, tf.zeros_like(token_weights))
    token_weights = utils.pad_to_max_len(token_weights, config.max_seq_len, -1)
    token_weights = tf.where(
        target_seq == vocab.PADDING_TOKEN,
        tf.zeros_like(token_weights) + config.eos_token_weight, token_weights)

    if training:
      return features['image'], input_seq, target_seq, token_weights
    else:
      return features['image'], response_seq, batched_examples

  def infer(self, model, preprocessed_outputs):
    """Perform  given the model and preprocessed outputs."""
    config = self.config.task
    mconfig = self.config.model
    image, _, examples = preprocessed_outputs  # response_seq unused by default
    if config.use_gt_box_at_test:  # Use gt bbox instead of predicted ones.
      encoded = None
      pred_classes = examples[1]['label']
      pred_bboxes = examples[1]['bbox']
      scores = tf.where(tf.greater(pred_classes, 0), examples[1]['scores'], 0.)
    else:
      bsz = tf.shape(image)[0]
      prompt_seq = task_utils.build_prompt_seq_from_task_id(
          config.object_detection_vocab_id, prompt_shape=(bsz, 1))
      pred_seq, logits, encoded = model.infer(
          image, prompt_seq,
          encoded=None,
          max_seq_len=(config.max_instances_per_image_test * 5 + 1),
          temperature=config.temperature,
          top_k=config.top_k, top_p=config.top_p)
      pred_classes, pred_bboxes, scores = task_utils.decode_object_seq_to_bbox(
          logits, pred_seq, config.quantization_bins, mconfig.coord_vocab_shift)
    prompt_seq = task_utils.build_instance_prompt_seq(
        self.task_vocab_id, pred_bboxes, pred_classes,
        config.quantization_bins, mconfig.coord_vocab_shift)

    pred_seq, _, _ = model.infer(  # (bsz * instances * num_samples, seqlen)
        image, prompt_seq,
        encoded=encoded, max_seq_len=config.max_points_per_object * 2 + 6,
        temperature=config.temperature, top_k=config.top_k, top_p=config.top_p,
        num_samples=config.ensemble_num_samples)

    # if True:  # Sanity check by using gt response_seq as pred_seq.
    #   pred_classes = examples[1]['label']
    #   pred_bboxes = examples[1]['bbox']
    #   scores = tf.cast(tf.greater(pred_classes, 0), tf.float32)
    #   pred_seq = utils.flatten_batch_dims(preprocessed_outputs[1], 2)

    pred_classes = tf.reshape(pred_classes, [-1])  # (bsz * instances)
    pred_bboxes = tf.reshape(pred_bboxes, [-1, 4])  # (bsz * instances, 4)
    scores = tf.reshape(scores, [-1])  # (bsz * instances)

    return examples, pred_seq, pred_classes, pred_bboxes, scores

  def postprocess_tpu(self, batched_examples, pred_seq, pred_classes,
                      pred_bboxes, scores, training=False):
    """Organizing results after fitting the batched examples in graph.

    Such as updating metrics, putting together results for computing metrics in
      CPU/numpy mode.

    Note: current implementation only support eval mode where gt are given in
      metrics as they are not constructed here from input_seq/target_seq.

    Args:
      batched_examples: a tupple of features (`dict`) and labels (`dict`),
        containing images and labels.
      pred_seq: `int` sequence of shape (bsz * instances * num_samples, seqlen).
      pred_classes: `int` of shape (bsz * instances).
      pred_bboxes: `float` of shape (bsz * instances', 4).
      scores: `float` of shape (bsz * instances).
      training: `bool` indicating training or  mode.

    Returns:
      results for passing to `postprocess_cpu` which runs in CPU mode.
    """
    config = self.config.task
    mconfig = self.config.model
    features, _ = batched_examples
    images, image_ids = features['image'], features['image/id']
    image_size_before_cropping = features['image_size_before_cropping']
    orig_image_size = features['orig_image_size']
    unpadded_image_size = features['unpadded_image_size']

    # Tile image related features to support multiple instances per image.
    bsz = tf.shape(images)[0]
    n_instances = tile_factor = tf.math.floordiv(
        tf.shape(pred_seq)[0], bsz * config.ensemble_num_samples)
    image_ids = utils.tile_along_batch(image_ids, tile_factor)

    # Tile the following features to support multiple samples per instance.
    tile_factor *= config.ensemble_num_samples
    image_size_before_cropping = utils.tile_along_batch(
        image_size_before_cropping, tile_factor)
    orig_image_size = utils.tile_along_batch(orig_image_size, tile_factor)
    unpadded_image_size = utils.tile_along_batch(unpadded_image_size,
                                                 tile_factor)

    # Compute coordinate scaling from [0., 1.] to actual pixel.
    image_size = images.shape[1:3].as_list()
    if training:
      # scale points to whole image size during train.
      scale = utils.tf_float32(image_size)
    else:
      # scale points to original image size during eval.
      scale = (
          utils.tf_float32(image_size)[tf.newaxis, :] /
          utils.tf_float32(unpadded_image_size))
      scale = scale * utils.tf_float32(orig_image_size)
      scale = tf.expand_dims(scale, 1)

    # Decode sequence output.
    pred_points = task_utils.decode_instance_seq_to_points(
        pred_seq, config.quantization_bins, mconfig.coord_vocab_shift)
    pred_points = pred_points[:, :config.max_points_per_object * 2]
    pred_points_rescaled = utils.scale_points(pred_points, scale)
    if 'crop_offset' in features:
      features['crop_offset'] = utils.tile_along_batch(
          features['crop_offset'], config.ensemble_num_samples)
      pred_points_rescaled = data_utils.adjust_for_crop_offset(
          pred_points_rescaled, features, config.max_points_per_object)
    return (images, image_ids, image_size_before_cropping, unpadded_image_size,
            [n_instances], pred_points, pred_points_rescaled, pred_classes,
            pred_bboxes, scores)

  def postprocess_cpu(self, outputs, train_step,
                      eval_step=None, training=False, summary_tag='eval',
                      ret_results=True):
    """CPU post-processing of outputs.

    Such as computing the metrics, log image summary.

    Note: current implementation only support eval mode where gt are given in
      metrics as they are not given here in outputs.

    Args:
      outputs: a tuple of tensor passed from `postprocess_tpu`.
      train_step: `int` scalar indicating training step of current model or
        the checkpoint.
      eval_step: `int` scalar indicating eval step for the given checkpoint.
      training: `bool` indicating training or  mode.
      summary_tag: `string` of name scope for result summary.
      ret_results: whether to return visualization images.

    Returns:
      A dict of visualization images if ret_results, else None.
    """
    # Copy outputs to cpu.
    new_outputs = []
    for i in range(len(outputs)):
      logging.info('Copying output at index %d to cpu for cpu post-process', i)
      new_outputs.append(tf.identity(outputs[i]))
    (images, image_ids, image_size_before_cropping, unpadded_image_size, n_instances,  # pylint: disable=unbalanced-tuple-unpacking
     pred_points, pred_points_rescaled, pred_classes, pred_bboxes,
     scores) = new_outputs
    n_instances = n_instances[0]
    num_samples = self.config.task.ensemble_num_samples
    threshold = self.config.task.ensemble_threshold
    bsz = tf.shape(images)[0]

    # Log/accumulate metrics.
    mask_size = unpadded_image_size if training else image_size_before_cropping
    pred_masks_rle = segment_to_mask_rle(
        pred_points_rescaled.numpy(), mask_size.numpy())
    # Ensemble mask for metrics.
    if self._coco_metrics:
      pred_masks_np = mask_rle_to_mask_np(pred_masks_rle, mask_size.numpy())
      pred_masks_np = ensemble_mask_np(
          pred_masks_np, (bsz * n_instances).numpy(), num_samples, threshold)
      self._coco_metrics.record_prediction(
          image_ids, pred_masks_np, pred_classes, scores)

    # Image summary.
    image_size = images.shape[1:3].as_list()
    pred_points_rescaled = utils.scale_points(pred_points, image_size)
    pred_masks_rle = segment_to_mask_rle(
        pred_points_rescaled.numpy(), image_size, is_single_shape=True)
    pred_masks_np = mask_rle_to_mask_np(
        pred_masks_rle, image_size, is_single_shape=True)
    # Ensemble masks for image summary.
    pred_masks_np = ensemble_mask_np(
        pred_masks_np, (bsz * n_instances).numpy(), num_samples, threshold)
    pred_masks_np = np.stack(pred_masks_np).astype(np.uint8)
    mask_new_shape = [pred_masks_np.shape[0]//n_instances, n_instances] + list(
        pred_masks_np.shape[1:])
    pred_masks_np = np.reshape(pred_masks_np, mask_new_shape)
    pred_bboxes = tf.reshape(pred_bboxes, [-1, n_instances, 4])
    pred_classes = tf.reshape(pred_classes, [-1, n_instances])
    scores = tf.reshape(scores, [-1, n_instances])
    if (eval_step <= 10) or ret_results:
      val_list = [(pred_bboxes, pred_masks_np, pred_classes, scores, 'pred')]
      ret_images = {}
      for bboxes_, masks_, classes_, scores_, tag_ in val_list:
        bboxes_ = bboxes_.numpy()
        classes_ = classes_.numpy()
        scores_ = scores_.numpy()
        tag = summary_tag + '/' + task_utils.join_if_not_none(
            [tag_, 'mask', eval_step], '_')
        images_ = np.copy(tf.image.convert_image_dtype(images, tf.uint8))
        ret_images[tag_] = add_image_summary_with_mask(
            images_, bboxes_, masks_, classes_, scores_, self._category_names,
            train_step, tag)

    logging.info('Done post-process')
    if ret_results:
      return ret_images

  def evaluate(self, summary_writer, step, eval_tag):
    """Evaluate results on accumulated outputs (after multiple infer steps).

    Args:
      summary_writer: the summary writer.
      step: current step.
      eval_tag: `string` name scope for eval result summary.

    Returns:
      result as a `dict`.
    """
    metrics = self.compute_scalar_metrics(step)
    with summary_writer.as_default():
      with tf.name_scope(eval_tag):
        self._log_metrics(metrics, step)
      summary_writer.flush()
    result_json_path = os.path.join(
        self.config.model_dir, eval_tag + 'cocoeval.pkl')
    if self._coco_metrics:
      tosave = {'dataset': self._coco_metrics.dataset,
                'detections': np.array(self._coco_metrics.detections)}
      with tf.io.gfile.GFile(result_json_path, 'wb') as f:
        pickle.dump(tosave, f)
    self.reset_metrics()
    return metrics

  def compute_scalar_metrics(self, step):
    """Returns a dict containing scalar metrics to log."""
    if self._coco_metrics:
      return self._coco_metrics.result(step)
    else:
      return {}

  def reset_metrics(self):
    """Reset states of metrics accumulators."""
    if self._coco_metrics:
      self._coco_metrics.reset_states()


def add_image_summary_with_mask(images, bboxes, masks, classes, scores,
                                category_names, step, tag):
  """Adds image summary with GT / predicted points."""
  new_images = []
  for image_, boxes_, masks_, classes_, scores_ in zip(images, bboxes, masks,
                                                       classes, scores):
    keep_indices = np.where(classes_ > 0)[0]
    image = vis_utils.visualize_boxes_and_labels_on_image_array(
        image=image_,
        boxes=boxes_[keep_indices],
        classes=classes_[keep_indices],
        scores=scores_[keep_indices],
        category_index=category_names,
        instance_masks=masks_[keep_indices],
        use_normalized_coordinates=True,
        max_boxes_to_draw=100)
    new_images.append(tf.image.convert_image_dtype(image, tf.float32))
  tf.summary.image(tag, new_images, step=step)
  return new_images


def _split_segment(pred_segment):
  """Splits predicted segment on separator."""
  result = [[]]
  for i in range(len(pred_segment) // 2):
    x_index = i * 2
    y_index = i * 2 + 1
    # TODO(srbs): Right now we decode only till the first occurence of the
    # padding token. Verify this is correct.
    # TODO(srbs): Should we raise an error if one coordinate is pad/separator
    # and the other is not?
    if (pred_segment[x_index] == vocab.PADDING_FLOAT or
        pred_segment[y_index] == vocab.PADDING_FLOAT):
      break
    if (pred_segment[x_index] == vocab.SEPARATOR_FLOAT or
        pred_segment[y_index] == vocab.SEPARATOR_FLOAT):
      result.append([])
    else:
      result[-1].extend([pred_segment[x_index], pred_segment[y_index]])
  return [l for l in result if l]


def segment_to_mask_rle(seg_flat, image_shape, is_single_shape=False):
  """Returns RLE encoded mask."""
  masks = []
  for i in range(len(seg_flat)):
    seg_split = _split_segment(utils.yx2xy(seg_flat[i]))
    h, w = image_shape if is_single_shape else image_shape[i]
    # TODO(srbs): Do not do this. Instead drop segments with len <=4 since those
    # are invalid segments anyway.
    # rle = mask_lib.frPyObjects(seg_split, h=h, w=w)
    rle = _mask.frPoly(seg_split, h=h, w=w)
    masks.append(mask_lib.merge(rle))
  return masks


def mask_rle_to_mask_np(masks, image_shape, is_single_shape=False):
  """Returns msak as np array."""
  decoded_masks = []
  for i, m in enumerate(masks):
    decoded_m = mask_lib.decode(m).astype(np.float32)
    shape = image_shape if is_single_shape else image_shape[i]
    if decoded_m.shape[0] == 0:
      decoded_m = np.zeros(shape)
    decoded_masks.append(decoded_m)
  if is_single_shape:
    return np.stack(decoded_masks).astype(np.uint8)
  else:  # return masks of multiple shapes as a list.
    return [d.astype(np.uint8) for d in decoded_masks]



def ensemble_mask_np(mask_nps, batch_size, num_samples, threshold=0.5):
  """Ensemble mask from multiple masks."""
  ensembled_masks_np = []
  for i in range(batch_size):
    for j in range(num_samples):
      ind = i * num_samples + j
      if j == 0:
        curr_mask = mask_nps[ind]
      else:
        curr_mask += mask_nps[ind]
    curr_mask = curr_mask / float(num_samples)
    curr_mask = np.greater_equal(curr_mask, threshold)
    ensembled_masks_np.append(curr_mask)
  return ensembled_masks_np
