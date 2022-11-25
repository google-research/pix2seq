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
"""COCO-style evaluation metrics.

COCO API: github.com/cocodataset/cocoapi/
"""

import copy
import io
import json
import os
import tempfile
import uuid
import zipfile

from absl import logging
import numpy as np
import PIL
import utils
import vocab
from metrics import metric_registry
from pycocotools import mask as mask_lib
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from skimage import segmentation
import tensorflow as tf


_CONFLICTING_ANNOTATIONS_ERROR = """
Manual annotations cannot be specified when coco_annotations_dir_for_metrics 
is set. To specify manual annotations use 
--config.dataset.coco_annotations_dir_for_metrics=''.
"""


class _CocoNonPanopticMetric():
  """Helper for non-panoptic metric types."""

  def __init__(self,
               config,
               iou_type):
    """Constructs COCO evaluation class.

    Args:
      config: The config.
      iou_type: One of 'segm', 'bbox' or 'keypoints'.
    """
    # Note: This can't directly be used to score predictions make using tfds
    # labels.
    self.metric_names = [
        'AP', 'AP_50', 'AP_75', 'AP_small', 'AP_medium', 'AP_large', 'AR_max_1',
        'AR_max_10', 'AR_max_100', 'AR_small', 'AR_medium', 'AR_large'
    ]
    self.dataset = {}
    gt_annotations_path = get_annotations_path_for_metrics(config)
    self.gt_annotations_path = gt_annotations_path
    self.coco_gt = COCO(gt_annotations_path)
    self.iou_type = iou_type
    self.filter_images_not_in_predictions = gt_annotations_path and bool(
        config.eval.steps)
    self.reset_states()

  def reset_states(self):
    """Reset COCO API object."""
    self.manual_dataset = {'images': [], 'annotations': [], 'categories': [],
                           'category_ids': []}
    self.annotation_id = 1
    self.detections = []
    self.metric_values = None

  def _filter_dataset(self, dataset, detections):
    """Returns new dataset only containing image ids in detections."""
    image_ids = set([dt['image_id'] for dt in detections])
    new_dataset = copy.copy(dataset)  # Shallow copy
    new_dataset['images'] = [
        img for img in new_dataset['images'] if img['id'] in image_ids
    ]
    new_dataset['annotations'] = [
        ann for ann in new_dataset['annotations']
        if ann['image_id'] in image_ids
    ]
    return new_dataset

  def _evaluate(self):
    """Evaluates with detections from all images with COCO API.

    Call this function from `self.result`.

    Returns:
      dict from metric name to float value
    """
    if not self.detections:
      logging.info('No valid detections, nothing to evaluate')
      return dict(
          zip(self.metric_names,
              np.zeros([len(self.metric_names)], dtype=np.float32)))
    dataset = copy.copy(self.coco_gt.dataset)
    # Use manual GT annotations if provided.
    if not self.gt_annotations_path:
      dataset = self.manual_dataset
      dataset['categories'] = [{
          'id': int(category_id)
      } for category_id in list(set(dataset['category_ids']))]
    if self.filter_images_not_in_predictions:
      # Only keep image_ids in self.detections.
      dataset = self._filter_dataset(dataset, self.detections)

    self.dataset = dataset
    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()

    # Run on validation dataset.
    coco_dt = coco_gt.loadRes(self.detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=self.iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats

    return dict(
        zip(self.metric_names, np.array(coco_metrics, dtype=np.float32)))

  def result(self, step):
    """Return the metric values (and compute it if needed)."""
    if self.metric_values is None:
      self.metric_values = self._evaluate()
    return self.metric_values


@metric_registry.MetricRegistry.register('coco_instance_segmentation')
class CocoInstanceSegmentationMetric(_CocoNonPanopticMetric):
  """COCO instance segmentation metric class."""

  def __init__(self, config):
    super().__init__(
        config,
        iou_type='segm')

  def record_prediction(self,
                        image_ids,
                        segments,
                        classes,
                        scores):
    """Records the detections and the corresponding gt data for a batch.

    Args:
      image_ids: [batch]
      segments: List of length `batch` containing bool np segmentation masks.
      classes: [batch] Predicted class normalized to [0, num_classes]. 0
        denotes padding.
      scores: [batch]
    """
    (image_ids, segments, classes, scores) = to_numpy(image_ids, segments,
                                                      classes, scores)
    segments = [
        mask_lib.encode(np.asfortranarray(curr_mask)) for curr_mask in segments
    ]
    batch_size = len(image_ids)
    for i in range(batch_size):
      if classes[i] != 0:
        dt = {}
        dt['image_id'] = image_ids[i]
        dt['category_id'] = classes[i]
        dt['segmentation'] = segments[i]
        dt['score'] = scores[i]
        self.detections.append(dt)

  def record_groundtruth(self,
                         image_ids,
                         segments,
                         classes,
                         areas):
    """Record GT annotations.

    Args:
      image_ids: [batch]
      segments: List of length `batch` containing bool np segmentation masks.
      classes: [batch]
      areas: [batch]
    """
    assert not self.gt_annotations_path, _CONFLICTING_ANNOTATIONS_ERROR
    (image_ids, segments, classes,
     areas) = to_numpy(image_ids, segments, classes, areas)
    segments = [
        mask_lib.encode(np.asfortranarray(curr_mask)) for curr_mask in segments
    ]
    batch_size = len(image_ids)
    for i in range(batch_size):
      if classes[i] != 0:
        self.manual_dataset['images'].append({
            'id': int(image_ids[i]),
        })
        self.manual_dataset['annotations'].append({
            'id': int(self.annotation_id),
            'image_id': image_ids[i],
            'category_id': classes[i],
            'segmentation': segments[i],
            'iscrowd': 0,
            'area': areas[i]
        })
        self.manual_dataset['category_ids'].append(int(classes[i]))
        self.annotation_id += 1


@metric_registry.MetricRegistry.register('coco_keypoint_detection')
class CocoKeypointDetectionMetric(_CocoNonPanopticMetric):
  """COCO keypoint detection metric class."""

  def __init__(self, config):
    super().__init__(
        config,
        iou_type='keypoints')

  def record_prediction(self, image_ids, keypoints, classes, scores):
    """Records the detections and the corresponding gt data for a batch.

    Args:
      image_ids: [batch]
      keypoints: [batch, 2*num_points]
      classes: [batch] Predicted class normalized to [0, num_classes]. 0
        denotes padding.
      scores: [batch]
    """
    (image_ids, keypoints, classes, scores) = to_numpy(image_ids, keypoints,
                                                       classes, scores)
    batch_size = len(image_ids)
    for i in range(batch_size):
      if classes[i] == 1:
        dt = {}
        dt['image_id'] = image_ids[i]
        dt['category_id'] = 1
        # The visibility flag is ignored here.
        dt['keypoints'] = insert_visibility(utils.yx2xy(keypoints[i]))
        dt['score'] = scores[i]
        self.detections.append(dt)

  def record_groundtruth(self,
                         image_ids,
                         keypoints, bbox, classes,
                         area, iscrowd, num_keypoints):
    """Record GT annotations.

    Args:
      image_ids: [batch]
      keypoints: [batch, 2*num_points]
      bbox: [bsz, 4]
      classes: [batch]
      area: [batch]
      iscrowd: [batch]
      num_keypoints: [batch]
    """
    assert not self.gt_annotations_path, _CONFLICTING_ANNOTATIONS_ERROR
    (image_ids, keypoints, bbox, classes, area, iscrowd,
     num_keypoints) = to_numpy(image_ids, keypoints, bbox, classes, area,
                               iscrowd, num_keypoints)
    batch_size = len(image_ids)
    for i in range(batch_size):
      if classes[i] != 0:
        self.manual_dataset['images'].append({
            'id': int(image_ids[i]),
        })
        assert classes[i] == 1
        self.manual_dataset['annotations'].append({
            'id': int(self.annotation_id),
            'image_id': image_ids[i],
            'category_id': 1,
            # TODO(srbs): Insert the actual GT visibility flag.
            'keypoints': insert_visibility(utils.yx2xy(keypoints[i])),
            'num_keypoints': num_keypoints[i],
            'bbox': yxyx_to_xywh(bbox[i]),
            'area': area[i],
            'iscrowd': iscrowd[i],
        })
        self.manual_dataset['category_ids'].append(1)
        self.annotation_id += 1


def insert_visibility(keypoints):
  """Insert the visibility flag.

  Args:
    keypoints: [2 * num_points]

  Returns:
    Array with length 3 * num_points.
  """
  def is_hidden(point):
    return any(i == vocab.INVISIBLE_FLOAT for i in point)

  new_list = []
  num_keypoints = len(keypoints) // 2
  for i in range(num_keypoints):
    point = keypoints[i * 2:(i + 1) * 2]
    new_list.extend(point)
    if is_hidden(point):
      new_list.append(0)
    else:
      new_list.append(1)
  return new_list


@metric_registry.MetricRegistry.register('coco_object_detection')
class CocoObjectDetectionMetric(_CocoNonPanopticMetric):
  """COCO object detection metric class."""

  def __init__(self, config):
    super().__init__(
        config,
        iou_type='bbox')

  def record_prediction(self,
                        image_ids,
                        pred_bboxes,
                        pred_classes,
                        pred_scores):
    """Records the detections and the corresponding gt data for a batch.

    Args:
      image_ids: [batch]
      pred_bboxes: [batch, instances, 4]
      pred_classes: [batch] Predicted class normalized to [0, num_classes]. 0
        denotes padding.
      pred_scores: [batch]
    """
    (image_ids, pred_bboxes, pred_classes,
     pred_scores) = to_numpy(image_ids, pred_bboxes, pred_classes, pred_scores)
    batch_size = pred_bboxes.shape[0]
    for i in range(batch_size):
      # Filter out predictions with predicted class label = 0.
      keep_indices = np.where(pred_classes[i] > 0)[0]
      for idx in keep_indices:
        box = pred_bboxes[i, idx]
        xmin, ymin, w, h = yxyx_to_xywh(box)
        self.detections.append({
            'image_id': int(image_ids[i]),
            'bbox': [xmin, ymin, w, h],
            'score': pred_scores[i, idx],
            'category_id': int(pred_classes[i, idx]),
        })

  def record_groundtruth(self,
                         image_ids,
                         bboxes,
                         classes,
                         areas,
                         is_crowds):
    """Record the groundtruth annotations.

    Args:
      image_ids: [batch]
      bboxes: [batch, instances, 4]
      classes: [batch]
      areas: [batch]
      is_crowds: [batch]
    """
    assert not self.gt_annotations_path, _CONFLICTING_ANNOTATIONS_ERROR
    (image_ids, bboxes, classes, areas,
     is_crowds) = to_numpy(image_ids, bboxes, classes, areas, is_crowds)
    batch_size = len(image_ids)
    for i in range(batch_size):
      self.manual_dataset['images'].append({
          'id': int(image_ids[i]),
      })
      keep_indices = np.where(classes[i] > 0)[0]
      for idx in keep_indices:
        box = bboxes[i, idx]
        is_crowd = is_crowds[i, idx] if is_crowds is not None else False
        self.manual_dataset['annotations'].append({
            'id': int(self.annotation_id),
            'image_id': int(image_ids[i]),
            'category_id': int(classes[i, idx]),
            'bbox': yxyx_to_xywh(box),
            'area': areas[i, idx],
            'iscrowd': is_crowd
        })
        self.manual_dataset['category_ids'].append(int(classes[i, idx]))
        self.annotation_id += 1


@metric_registry.MetricRegistry.register('coco_captioning')
class CocoCaptioningEvaluationMetric():
  """COCO evaluation for image captioning."""

  def __init__(self, config):
    self.metric_names = ['BLEU-4', 'CIDEr', 'ROUGE_L']
    self.gt_annotations_path = get_annotations_path_for_metrics(
        config)
    self.result_dir = config.model_dir
    self.reset_states()

  def reset_states(self):
    """Reset COCO API object."""
    self.captions = []
    self.metric_values = None

  def record_prediction(self, image_ids, captions, original_pred, clipped_pred):
    """Records the captions for a batch."""
    for i, c, o, p in zip(image_ids, captions, original_pred, clipped_pred):
      self.captions.append({
          'image_id': int(i),
          'caption': c,
          'original_pred': [int(v) for v in o],
          'clipped_pred': [int(v) for v in p],
      })

  def _evaluate(self, step):
    """Evaluates with captions from all images with COCO API.

    Call this function from `self.result`.

    Args:
      step: int. The checkpoint step being evaluated on.

    Returns:
      dict from metric name to float value
    """
    result_file = self._write_result_file(step)
    return dict(
        zip(self.metric_names,
            np.zeros([len(self.metric_names)], dtype=np.float32)))

  def _write_result_file(self, step):
    result_file = None
    if self.captions and self.result_dir:
      result_file = os.path.join(self.result_dir,
                                 f'coco_result_{step}_{uuid.uuid4()}.json')
      with tf.io.gfile.GFile(result_file, 'w') as f:
        json.dump(self.captions, f)
    return result_file

  def result(self, step):
    """Return the metric values (and compute it if needed)."""
    if self.metric_values is None:
      self.metric_values = self._evaluate(step)
    return self.metric_values


def get_annotations_path_for_metrics(config):  # pylint: disable=missing-function-docstring
  annotations_dir = config.dataset.get('coco_annotations_dir_for_metrics')
  if not annotations_dir:
    return None
  split = config.dataset.train_split if config.training else (
      config.dataset.eval_split)
  filename = (
      config.dataset.get('train_filename_for_metrics')
      if split == 'train' else config.dataset.get('val_filename_for_metrics'))

  return os.path.join(annotations_dir, filename)


def to_numpy(*args):
  results = tuple(arg.numpy() if hasattr(arg, 'numpy') else arg for arg in args)
  # batch_sizes = set([len(arr) for arr in results if arr is not None])
  # assert len(batch_sizes) == 1
  return results


# TODO(srbs): Remove this after updating other users.
def yxyx_to_xywh(box):
  ymin, xmin, ymax, xmax = box
  w = xmax - xmin
  h = ymax - ymin
  return [xmin, ymin, w, h]
