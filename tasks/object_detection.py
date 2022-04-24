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
"""Object detection task via COCO metric evaluation."""

import os
import pickle

from absl import logging
import ml_collections
import numpy as np
import utils
import vocab
from data import data_utils
from metrics import coco_metrics
from tasks import task as task_lib
from tasks import task_utils
from tasks.visualization import vis_utils
import tensorflow as tf


@task_lib.TaskRegistry.register('object_detection')
class TaskObjectDetection(task_lib.Task):
  """Object detection task with coco metric evaluation."""

  def __init__(self,
               config: ml_collections.ConfigDict):
    super().__init__(config)

    if config.task.get('max_seq_len', 'auto') == 'auto':
      self.config.task.max_seq_len = config.task.max_instances_per_image * 5
    anno = task_utils.coco_annotation_path(config, ret_category_names=True)
    self._category_names = anno['category_names']
    self._coco_metrics = coco_metrics.CocoObjectDetectionMetric(
        gt_annotations_path=anno['gt_annotations_path'],
        filter_images_not_in_predictions=(anno['gt_annotations_path'] and
                                          config.eval.steps is not None))

  def preprocess_single(self, dataset, batch_duplicates, training):
    """Task-specific preprocessing of individual example in the dataset.

    Typical operations in this preprocessing step for detection task:
      - Image augmentation such random resize & cropping, color jittering, flip.
      - Label augmentation such as sampling noisy & duplicated boxes.

    Args:
      dataset: A tf.data.Dataset.
      batch_duplicates: `int`, enlarge a batch by augmenting it multiple times
        (as specified) and concating the augmented examples.
      training: bool.

    Returns:
      A dataset.
    """

    def _preprocess_single_example(features, labels):
      config = self.config.task
      features['orig_image_size'] = tf.shape(features['image'])[:2]

      if training:
        features_list, labels_list = [], []
        for _ in range(batch_duplicates):
          features_, labels_ = data_utils.preprocess_train(
              features,
              labels,
              max_image_size=config.image_size,
              max_instances_per_image=config.max_instances_per_image,
              object_order=config.object_order,
              inject_noise_instances=config.noise_bbox_weight > 0,
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
            max_instances_per_image=config.max_instances_per_image,
            object_coordinate_keys=('bbox', 'polygon', 'keypoints'))

      return features, labels

    if training:
      dataset = dataset.filter(  # Filter out images with no annotations.
          lambda features, labels: tf.shape(labels['label'])[0] > 0)
    dataset = dataset.map(_preprocess_single_example,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def preprocess_batched(self, batched_examples, training):
    """Task-specific preprocessing of batched examples on accelerators (TPUs).

    Typical operations in this preprocessing step for detection task:
      - Quantization and serialization of object instances.
      - Creating the input sequence, target sequence, and token weights.

    Args:
      batched_examples: tuples of feature and label tensors that are
        preprocessed, batched, and stored with `dict`.
      training: bool.

    Returns:
      images: `float` of shape (bsz, h, w, c)
      input_seq: `int` of shape (bsz, seqlen).
      target_seq: `int` of shape (bsz, seqlen).
      token_weights: `float` of shape (bsz, seqlen).
    """
    config = self.config.task
    mconfig = self.config.model
    features, labels = batched_examples

    # Create input/target seq.
    ret = build_response_seq_from_bbox(
        labels['bbox'], labels['label'], config.quantization_bins,
        config.noise_bbox_weight, mconfig.coord_vocab_shift,
        class_label_corruption=config.class_label_corruption)
    response_seq, response_seq_cm, token_weights = ret
    prompt_seq = task_utils.build_prompt_seq_from_task_id(
        self.task_vocab_id, response_seq)  # (bsz, 1)
    input_seq = tf.concat([prompt_seq, response_seq_cm], -1)
    target_seq = tf.concat([prompt_seq, response_seq], -1)

    # Pad sequence to a unified maximum length.
    assert input_seq.shape[-1] <= config.max_seq_len + 1
    input_seq = utils.pad_to_max_len(input_seq, config.max_seq_len + 1, -1)
    target_seq = utils.pad_to_max_len(target_seq, config.max_seq_len + 1, -1)
    input_seq, target_seq = input_seq[..., :-1], target_seq[..., 1:]
    token_weights = utils.pad_to_max_len(token_weights, config.max_seq_len, -1)

    # Assign lower weights for ending/padding tokens.
    token_weights = tf.where(
        target_seq == vocab.PADDING_TOKEN,
        tf.zeros_like(token_weights) + config.eos_token_weight, token_weights)

    if training:
      return features['image'], input_seq, target_seq, token_weights
    else:
      return features['image'], response_seq, batched_examples

  def infer(self, model, preprocessed_outputs):
    """Perform inference given the model and preprocessed outputs."""
    config = self.config.task
    image, _, examples = preprocessed_outputs  # response_seq unused by default
    bsz = tf.shape(image)[0]
    prompt_seq = task_utils.build_prompt_seq_from_task_id(
        self.task_vocab_id, prompt_shape=(bsz, 1))
    pred_seq, logits, _ = model.infer(
        image, prompt_seq, encoded=None,
        max_seq_len=(config.max_instances_per_image_test * 5 + 1),
        temperature=config.temperature, top_k=config.top_k, top_p=config.top_p)
    # if True:  # Sanity check by using gt response_seq as pred_seq.
    #   pred_seq = preprocessed_outputs[1]
    #   logits = tf.one_hot(pred_seq, mconfig.vocab_size)
    return examples, pred_seq, logits

  def postprocess_tpu(self, batched_examples, pred_seq, logits, training=False):
    """Organizing results after fitting the batched examples in graph.

    Such as updating metrics, putting together results for computing metrics in
      CPU/numpy mode.

    Note: current implementation only support eval mode where gt are given in
      metrics as they are not constructed here from input_seq/target_seq.

    Args:
      batched_examples: a tupple of features (`dict`) and labels (`dict`),
        containing images and labels.
      pred_seq: `int` sequence of shape (bsz, seqlen').
      logits: `float` sequence of shape (bsz, seqlen', vocab_size).
      training: `bool` indicating training or inference mode.

    Returns:
      results for passing to `postprocess_cpu` which runs in CPU mode.
    """
    config = self.config.task
    mconfig = self.config.model
    features, labels = batched_examples
    images, image_ids = features['image'], features['image/id']
    orig_image_size = features['orig_image_size']
    unpadded_image_size = features['unpadded_image_size']

    # Decode sequence output.
    pred_classes, pred_bboxes, scores = task_utils.decode_object_seq_to_bbox(
        logits, pred_seq, config.quantization_bins, mconfig.coord_vocab_shift)

    # Compute coordinate scaling from [0., 1.] to actual pixels in orig image.
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
    pred_bboxes_rescaled = utils.scale_points(pred_bboxes, scale)

    gt_classes, gt_bboxes = labels['label'], labels['bbox']
    gt_bboxes_rescaled = utils.scale_points(gt_bboxes, scale)
    area, is_crowd = labels['area'], labels['is_crowd']

    return (images, image_ids, pred_bboxes, pred_bboxes_rescaled, pred_classes,
            scores, gt_classes, gt_bboxes, gt_bboxes_rescaled, area, is_crowd)

  def postprocess_cpu(self, outputs, train_step,
                      eval_step=None, training=False, summary_tag='eval',
                      ret_results=False):
    """CPU post-processing of outputs.

    Such as computing the metrics, log image summary.

    Note: current implementation only support eval mode where gt are given in
      metrics as they are not given here in outputs.

    Args:
      outputs: a tuple of tensor passed from `postprocess_tpu`.
      train_step: `int` scalar indicating training step of current model or
        the checkpoint.
      eval_step: `int` scalar indicating eval step for the given checkpoint.
      training: `bool` indicating training or inference mode.
      summary_tag: `string` of name scope for result summary.
      ret_results: whether to return visualization images.

    Returns:
      A dict of visualization images if ret_results, else None.
    """
    # Copy outputs to cpu.
    new_outputs = []
    for i in range(len(outputs)):
      logging.info('Copying output at index %d to cpu', i)
      new_outputs.append(tf.identity(outputs[i]))
    (images, image_ids, pred_bboxes, pred_bboxes_rescaled, pred_classes,  # pylint: disable=unbalanced-tuple-unpacking
     scores, gt_classes, gt_bboxes, gt_bboxes_rescaled, area, is_crowd
     ) = new_outputs

    # Log/accumulate metrics.
    if self._coco_metrics.gt_annotations_path:
      self._coco_metrics.record_detection(
          image_ids, pred_bboxes_rescaled, pred_classes, scores)
    else:
      self._coco_metrics.record_detection(
          image_ids, pred_bboxes_rescaled, pred_classes, scores,
          gt_bboxes_rescaled, gt_classes, areas=area, is_crowds=is_crowd)

    # Image summary.
    if eval_step <= 10 or ret_results:
      image_ids_ = image_ids.numpy()
      gt_tuple = (gt_bboxes, gt_classes, scores * 0. + 1., 'gt')  # pylint: disable=unused-variable
      pred_tuple = (pred_bboxes, pred_classes, scores, 'pred')
      vis_list = [pred_tuple]  # exclude gt for simplicity.
      ret_images = {}
      for bboxes_, classes_, scores_, tag_ in vis_list:
        tag = summary_tag + '/' + task_utils.join_if_not_none(
            [tag_, 'bbox', eval_step], '_')
        bboxes_, classes_, scores_ = (
            bboxes_.numpy(), classes_.numpy(), scores_.numpy())
        images_ = np.copy(tf.image.convert_image_dtype(images, tf.uint8))
        ret_images[tag_] = add_image_summary_with_bbox(
            images_, bboxes_, classes_, scores_, self._category_names,
            image_ids_, train_step, tag,
            max_images_shown=(-1 if ret_results else 3))

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
    tosave = {'dataset': self._coco_metrics.dataset,
              'detections': np.array(self._coco_metrics.detections)}
    with tf.io.gfile.GFile(result_json_path, 'wb') as f:
      pickle.dump(tosave, f)
    self.reset_metrics()
    return metrics

  def compute_scalar_metrics(self, step):
    """Returns a dict containing scalar metrics to log."""
    return self._coco_metrics.result()

  def reset_metrics(self):
    """Reset states of metrics accumulators."""
    self._coco_metrics.reset_states()


def add_image_summary_with_bbox(images, bboxes, classes, scores, category_names,
                                image_ids, step, tag, max_images_shown=3):
  """Adds image summary with GT / predicted bbox."""
  k = 0
  del image_ids
  new_images = []
  for image_, boxes_, scores_, classes_ in zip(images, bboxes, scores, classes):
    keep_indices = np.where(classes_ > 0)[0]
    image = vis_utils.visualize_boxes_and_labels_on_image_array(
        image=image_,
        boxes=boxes_[keep_indices],
        classes=classes_[keep_indices],
        scores=scores_[keep_indices],
        category_index=category_names,
        use_normalized_coordinates=True,
        min_score_thresh=0.1,
        max_boxes_to_draw=100)
    new_images.append(tf.image.convert_image_dtype(image, tf.float32))
    k += 1
    if max_images_shown >= 0 and k >= max_images_shown:
      break
  tf.summary.image(tag, new_images, step=step, max_outputs=max_images_shown)
  return new_images


def build_response_seq_from_bbox(bbox,
                                 label,
                                 quantization_bins,
                                 noise_bbox_weight,
                                 coord_vocab_shift,
                                 class_label_corruption='rand_cls'):
  """"Build target seq from bounding bboxes for object detection.

  Objects are serialized using the format of yxyxc.

  Args:
    bbox: `float` bounding box of shape (bsz, n, 4).
    label: `int` label of shape (bsz, n).
    quantization_bins: `int`.
    noise_bbox_weight: `float` on the token weights for noise bboxes.
    coord_vocab_shift: `int`, shifting coordinates by a specified integer.
    class_label_corruption: `string` specifying how labels are corrupted for the
      input_seq.

  Returns:
    discrete sequences with shape (bsz, seqlen).
  """
  # Bbox and label quantization.
  is_padding = tf.expand_dims(tf.equal(label, 0), -1)
  quantized_bbox = utils.quantize(bbox, quantization_bins)
  quantized_bbox = quantized_bbox + coord_vocab_shift
  quantized_bbox = tf.where(is_padding,
                            tf.zeros_like(quantized_bbox), quantized_bbox)
  new_label = tf.expand_dims(label + vocab.BASE_VOCAB_SHIFT, -1)
  new_label = tf.where(is_padding, tf.zeros_like(new_label), new_label)
  lb_shape = tf.shape(new_label)

  # Bbox and label serialization.
  response_seq = tf.concat([quantized_bbox, new_label], axis=-1)
  response_seq = utils.flatten_non_batch_dims(response_seq, 2)
  rand_cls = vocab.BASE_VOCAB_SHIFT + tf.random.uniform(
      lb_shape,
      0,
      coord_vocab_shift - vocab.BASE_VOCAB_SHIFT,
      dtype=new_label.dtype)
  fake_cls = vocab.FAKE_CLASS_TOKEN + tf.zeros_like(new_label)
  rand_n_fake_cls = tf.where(
      tf.random.uniform(lb_shape) > 0.5, rand_cls, fake_cls)
  real_n_fake_cls = tf.where(
      tf.random.uniform(lb_shape) > 0.5, new_label, fake_cls)
  real_n_rand_n_fake_cls = tf.where(
      tf.random.uniform(lb_shape) > 0.5, new_label, rand_n_fake_cls)
  label_mapping = {'none': new_label,
                   'rand_cls': rand_cls,
                   'real_n_fake_cls': real_n_fake_cls,
                   'rand_n_fake_cls': rand_n_fake_cls,
                   'real_n_rand_n_fake_cls': real_n_rand_n_fake_cls}
  new_label_m = label_mapping[class_label_corruption]
  new_label_m = tf.where(is_padding, tf.zeros_like(new_label_m), new_label_m)
  response_seq_class_m = tf.concat([quantized_bbox, new_label_m], axis=-1)
  response_seq_class_m = utils.flatten_non_batch_dims(response_seq_class_m, 2)

  # Get token weights.
  is_real = tf.cast(tf.not_equal(new_label, vocab.FAKE_CLASS_TOKEN), tf.float32)
  bbox_weight = tf.tile(is_real, [1, 1, 4])
  label_weight = is_real + (1. - is_real) * noise_bbox_weight
  token_weights = tf.concat([bbox_weight, label_weight], -1)
  token_weights = utils.flatten_non_batch_dims(token_weights, 2)

  return response_seq, response_seq_class_m, token_weights
