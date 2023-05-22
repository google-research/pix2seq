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
"""Panoptic segmentation task via COCO metric evaluation."""

from absl import logging
import ml_collections
import numpy as np

import utils
from metrics import metric_registry
from tasks import task as task_lib
from tasks import task_utils
import tensorflow as tf


@task_lib.TaskRegistry.register('panoptic_segmentation')
class TaskPanopticSegmentation(task_lib.Task):
  """Panoptic segmentation with coco metric evaluation."""

  def __init__(self,
               config: ml_collections.ConfigDict):
    super().__init__(config)
    self._metrics = {
        'mean_iou': tf.keras.metrics.MeanIoU(
            config.dataset.num_classes, 'mean_iou'),
    }
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

    dataset = dataset.map(
        lambda x: self.preprocess_single_example(x, training, batch_duplicates),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def preprocess_batched(self, batched_examples, training):
    """Task-specific preprocessing of batched examples on accelerators (TPUs).

    Args:
      batched_examples: tuples of feature and label tensors that are
        preprocessed, batched, and stored with `dict`.
      training: bool.

    Returns:
      images: `float` of shape (bsz, h, w, c).
      label_map: `int32` of shape (bsz, h, w, 2).
    """
    t_cfg = self.config.task
    m_cfg = self.config.model

    # Obtain mask weight and convert masks into bits.
    mask_weight = task_utils.get_normalized_weight(
        batched_examples['label_map'][..., 1],
        t_cfg.max_instances_per_image,
        m_cfg.mask_weight_p)
    label_map = task_utils.integer_map_to_bits(
        batched_examples['label_map'], t_cfg.n_bits_label, m_cfg.b_scale)
    if training:
      return batched_examples['image'], label_map, mask_weight
    else:
      return batched_examples['image'], label_map, batched_examples

  def infer(self, model, preprocessed_outputs):
    """Perform  given the model and preprocessed outputs."""
    t_cfg = self.config.task
    m_cfg = self.config.model
    images, label_map, examples = preprocessed_outputs
    masks_pred = label_map  # comment `model.infer` for sanity check
    masks_pred = model.infer(images, m_cfg.iterations, m_cfg.sampler)
    masks_pred = task_utils.bits_to_panoptic_map(
        masks_pred, t_cfg.n_bits_label, self.config.dataset.num_classes,
        t_cfg.max_instances_per_image)

    if m_cfg.image_size != m_cfg.msize:
      masks_pred = tf.image.resize(
          masks_pred, m_cfg.image_size, method='nearest')
    return examples, masks_pred

  def postprocess_tpu(self, batched_examples, predictions, training=False):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Organizing results after fitting the batched examples in graph.

    Such as updating metrics, putting together results for computing metrics in
      CPU/numpy mode.

    Args:
      batched_examples: a tupple of features (`dict`) and labels (`dict`),
        containing images and labels.
      predictions: `int32` of shape (bsz, h, w, 2).
      training: bool.

    Returns:
      results for passing to `postprocess_cpu` which runs in CPU mode.
    """
    images, image_ids = batched_examples['image'], batched_examples['image/id']

    self._metrics['mean_iou'].update_state(
        batched_examples['label_map'][..., 0], predictions[..., 0])
    label_map_orig = batched_examples.get('label_map_orig',
                                          batched_examples['label_map'])
    return (images, image_ids, batched_examples['orig_image_size'],
            batched_examples['unpadded_image_size'], predictions,
            label_map_orig)

  def postprocess_cpu(self, outputs, train_step,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                      eval_step=None, training=False, summary_tag='eval',
                      ret_results=False):
    """CPU post-processing of outputs.

    Such as computing the metrics, log image summary.

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
    images, image_ids, orig_image_sizes, unpadded_image_sizes, predictions, gts = new_outputs  # pylint: disable=unbalanced-tuple-unpacking

    # Log/accumulate metrics.
    self._update_metrics(image_ids, predictions, gts, orig_image_sizes,
                         unpadded_image_sizes,
                         min_pixels=self.config.task.min_pixels)

    # Image summary.
    preds_semantic = utils.colorize(predictions[..., 0])
    preds_identity = utils.colorize(predictions[..., 1])
    # gts_semantic = utils.colorize(gts[..., 0])
    # gts_identity = utils.colorize(gts[..., 1])

    if (eval_step <= 5) or ret_results:
      val_list = [(preds_semantic, preds_identity, 'pred')]
      # val_list += [(gts_semantic, gts_identity, 'gt')]
      ret_images = {}
      for semantic, identity, tag_ in val_list:
        new_images = tf.concat([images, semantic, identity], 1)
        tag = summary_tag + '/' + task_utils.join_if_not_none(
            [tag_, 'mask', eval_step], '_')
        new_images = np.copy(tf.image.convert_image_dtype(new_images, tf.uint8))
        tf.summary.image(tag, new_images, step=train_step)
        ret_images[tag_] = new_images
    logging.info('Done post-process')
    if ret_results:
      return ret_images

  def _update_metrics(self, image_ids, predictions, gts, orig_image_sizes,
                      unpadded_image_sizes, min_pixels=0):
    if not self._coco_metrics:
      return
    preds_semantic = predictions[..., 0]
    preds_identity = predictions[..., 1]
    if not self._coco_metrics.gt_annotations_path:
      gts_semantic = gts[..., 0]
      gts_identity = gts[..., 1]
      gt_labels = []
      gt_instance_ids = []
    batch_size = len(preds_semantic)
    pred_labels = []
    pred_instance_ids = []
    for i in range(batch_size):
      unpadded_h, unpadded_w = unpadded_image_sizes[i].numpy()
      orig_h, orig_w = orig_image_sizes[i].numpy()
      # Remove padding
      pred_semantic = preds_semantic[i][:unpadded_h, :unpadded_w, tf.newaxis]
      pred_identity = preds_identity[i][:unpadded_h, :unpadded_w, tf.newaxis]
      if not self._coco_metrics.gt_annotations_path:
        gt_semantic = gts_semantic[i][:orig_h, :orig_w].numpy()
        gt_identity = gts_identity[i][:orig_h, :orig_w].numpy()
        gt_labels.append(gt_semantic)
        gt_instance_ids.append(gt_identity)
      # Scale to original size.
      pred_semantic = tf.image.resize(
          pred_semantic, [orig_h, orig_w],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[:, :, 0].numpy()
      pred_identity = tf.image.resize(
          pred_identity, [orig_h, orig_w],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[:, :, 0].numpy()
      pred_semantic, pred_identity = majority_vote_postprocess(
          pred_semantic, pred_identity, min_pixels,
          max_instances_per_image=self.config.task.max_instances_per_image)
      if self._coco_metrics.gt_annotations_path:
        # TODO(srbs): Use coco labels instead of tfds labels in input pipeline
        # so that we don't need to do this mapping from tfds to coco labels.
        # Class ids are shifted by 1 in the dataset since 0 is reserved for
        # unlabelled pixels.
        pred_semantic = np.maximum(pred_semantic - 1, 0)
        # Convert tfds labels to coco labels.
        def get_coco_labels(semantic, identity):
          coco_labels = tf.gather(self._coco_metrics.tfds_labels_to_coco,
                                  semantic)
          coco_labels = tf.where(tf.equal(identity, 0), 0, coco_labels)
          return coco_labels.numpy()
        pred_semantic = get_coco_labels(pred_semantic, pred_identity)
      pred_labels.append(pred_semantic)
      pred_instance_ids.append(pred_identity)

    self._coco_metrics.record_prediction(image_ids.numpy(), pred_labels,
                                         pred_instance_ids)
    if not self._coco_metrics.gt_annotations_path:
      self._coco_metrics.record_groundtruth(image_ids.numpy(), gt_labels,
                                            gt_instance_ids)

  def compute_scalar_metrics(self, step):
    """Returns a dict containing scalar metrics to log."""
    result = {}
    for metric in self._metrics.values():
      result[metric.name] = metric.result().numpy()

    if self._coco_metrics:
      result.update(self._coco_metrics.result(step))
    return result

  def reset_metrics(self):
    """Reset states of metrics accumulators."""
    for metric in self._metrics.values():
      metric.reset_states()
    if self._coco_metrics:
      self._coco_metrics.reset_states()


def majority_vote_postprocess(semantic_map,
                              identity_map,
                              min_pixels,
                              max_instances_per_image,
                              max_classes=300):
  """Returns a semantic map based on majority vote of identity ids."""
  assert semantic_map.shape == identity_map.shape
  h, w = semantic_map.shape
  indices = tf.concat(
      [tf.expand_dims(identity_map, -1),
       tf.expand_dims(semantic_map, -1)],
      axis=-1)
  indices = tf.reshape(indices, [h * w, 2])
  # Note: Here we use the fact that multiple updates at the same location are
  # summed.
  counts = tf.scatter_nd(indices, tf.ones([h * w], tf.int32),
                         [max_instances_per_image, max_classes])
  class_assignments = tf.argmax(counts, axis=-1)

  indices = tf.reshape(identity_map, [h * w, 1])
  pix_counts = tf.scatter_nd(indices, tf.ones([h * w], tf.int32),
                             [max_instances_per_image])

  class_assignments = tf.where(
      tf.greater_equal(pix_counts, min_pixels), class_assignments, 0)
  new_semantic_map = tf.gather(class_assignments, identity_map)
  new_identity_map = tf.where(tf.equal(new_semantic_map, 0), 0, identity_map)
  return new_semantic_map.numpy(), new_identity_map.numpy()
