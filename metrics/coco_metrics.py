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
import zipfile
from absl import logging
import numpy as np
import PIL

import utils
import vocab
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tensorflow as tf


class _CocoNonPanopticMetric():
  """Helper for non-panoptic metric types."""

  def __init__(self,
               gt_annotations_path,
               filter_images_not_in_predictions,
               iouType):
    """Constructs COCO evaluation class.

    Args:
      gt_annotations_path: Json containing GT annotations.
      filter_images_not_in_predictions: Whether to filters out images from the
        GT/annotation dataset for which there is no prediction. Set it to True
        if evaluating over only a subset of images in GT.
      iouType: One of 'segm', 'bbox' or 'keypoints'.
    """
    # Note: This can't directly be used to score predictions make using tfds
    # labels.
    self.metric_names = [
        'AP', 'AP_50', 'AP_75', 'AP_small', 'AP_medium', 'AP_large', 'AR_max_1',
        'AR_max_10', 'AR_max_100', 'AR_small', 'AR_medium', 'AR_large'
    ]
    self.dataset = {}
    self.gt_annotations_path = gt_annotations_path
    self.coco_gt = COCO(gt_annotations_path)
    self.iou_type = iouType
    self.filter_images_not_in_predictions = filter_images_not_in_predictions
    if gt_annotations_path is None:
      assert not filter_images_not_in_predictions, (
          'Do not support filtering without an annotation file.')
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

  def result(self):
    """Return the metric values (and compute it if needed)."""
    if self.metric_values is None:
      self.metric_values = self._evaluate()
    return self.metric_values


class CocoObjectDetectionMetric(_CocoNonPanopticMetric):
  """COCO object detection metric class."""

  def __init__(self,
               gt_annotations_path,
               filter_images_not_in_predictions=False):
    super().__init__(
        gt_annotations_path,
        filter_images_not_in_predictions=filter_images_not_in_predictions,
        iouType='bbox')

  def record_detection(self,
                       image_ids,
                       pred_bboxes,
                       pred_classes,
                       pred_scores,
                       gt_bboxes=None,
                       gt_classes=None,
                       areas=None,
                       is_crowds=None):
    """Records the detections and the corresponding gt data for a batch.

    Args:
      image_ids: [batch]
      pred_bboxes: [batch, instances, 4]
      pred_classes: [batch] Predicted class normalized to [0, num_classes]. 0
        denotes padding.
      pred_scores: [batch]
      gt_bboxes: [batch, instances, 4]
      gt_classes: [batch]
      areas: [batch]
      is_crowds: [batch]
    """
    (pred_bboxes, gt_bboxes, pred_classes, gt_classes, areas, pred_scores,
    ) = to_numpy(pred_bboxes, gt_bboxes, pred_classes,
                 gt_classes, areas, pred_scores)
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

    if gt_classes is not None:
      for i in range(batch_size):
        self.manual_dataset['images'].append({
            'id': int(image_ids[i]),
        })
        keep_indices = np.where(gt_classes[i] > 0)[0]
        for idx in keep_indices:
          box = gt_bboxes[i, idx]
          is_crowd = is_crowds[i, idx] if is_crowds is not None else False
          self.manual_dataset['annotations'].append({
              'id': int(self.annotation_id),
              'image_id': int(image_ids[i]),
              'category_id': int(gt_classes[i, idx]),
              'bbox': yxyx_to_xywh(box),
              'area': areas[i, idx],
              'iscrowd': is_crowd
          })
          self.manual_dataset['category_ids'].append(int(gt_classes[i, idx]))
          self.annotation_id += 1


def to_numpy(*args):
  results = tuple(arg.numpy() if hasattr(arg, 'numpy') else arg for arg in args)
  # batch_sizes = set([len(arr) for arr in results if arr is not None])
  # assert len(batch_sizes) == 1
  return results


def yxyx_to_xywh(box):
  ymin, xmin, ymax, xmax = box
  w = xmax - xmin
  h = ymax - ymin
  return [xmin, ymin, w, h]
