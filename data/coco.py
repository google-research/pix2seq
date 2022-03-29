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
"""MSCOCO/2017 datasets.

The coco/2017 ds available in tfds only has bbox information (as of April 2021)
but not instance segmentation or keypoints. The datasets here loads the
segmentation and keypoints annotation from raw coco jsons and join those with
the images in coco/2017 using the object/id. For keypoints dataset, instances
which are not in the person category are dropped.
"""

import collections
import json
import os
from absl import logging
import ml_collections
import numpy as np

import utils
import vocab
from data import dataset as dataset_lib
import tensorflow as tf


class CocoDataset(dataset_lib.Dataset):
  """Dataset for COCO tasks."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self._train_file_name = (
        self.config.train_filename if self.config.train_split == 'train'
        else self.config.val_filename)
    self._val_file_name = (
        self.config.train_filename if self.config.eval_split == 'train'
        else self.config.val_filename)

  @property
  def _train_annotations(self):
    if not hasattr(self, '_train_annotations_cache'):
      filename = self._train_file_name
      train_annotations_path = os.path.join(self.config.coco_annotations_dir,
                                            filename)
      self._train_annotations_cache = load_annnotations(
          train_annotations_path)
    return self._train_annotations_cache

  @property
  def _val_annotations(self):
    if not hasattr(self, '_val_annotations_cache'):
      filename = self._val_file_name
      val_annotations_path = os.path.join(self.config.coco_annotations_dir,
                                          filename)
      self._val_annotations_cache = load_annnotations(
          val_annotations_path)
    return self._val_annotations_cache

  @property
  def _train_id_to_ann(self):
    if not hasattr(self, '_train_id_to_ann_cache'):
      self._train_id_to_ann_cache = {
          ann['id']: ann for ann in self._train_annotations['annotations']
      }
    return self._train_id_to_ann_cache

  @property
  def _val_id_to_ann(self):
    if not hasattr(self, '_val_id_to_ann_cache'):
      self._val_id_to_ann_cache = {
          ann['id']: ann for ann in self._val_annotations['annotations']
      }
    return self._val_id_to_ann_cache

  def _get_areas(self, object_ids, training):
    num_instances = tf.shape(object_ids)[0]
    out_shape = [num_instances]
    out_dtype = tf.float32
    id_to_ann = (self._train_id_to_ann if training else self._val_id_to_ann)

    def get_area(ids):
      return np.asarray([id_to_ann[i]['area'] for i in ids], dtype=np.float32)

    areas = tf.numpy_function(get_area, (object_ids,), (out_dtype,))
    areas = tf.reshape(areas, out_shape)
    return areas

  def _get_labels(self, object_ids, training):
    """"Returns COCO category id in in [1, 91]."""
    num_instances = tf.shape(object_ids)[0]
    out_shape = [num_instances]
    out_dtype = tf.int64
    id_to_ann = (self._train_id_to_ann if training else self._val_id_to_ann)

    def get_label(ids):

      def id_to_label(i):
        if i in id_to_ann:
          return id_to_ann[i]['category_id']
        else:
          return 0

      return np.asarray([id_to_label(i) for i in ids], dtype=np.int64)

    label = tf.numpy_function(get_label, (object_ids,), (out_dtype,))
    label = tf.reshape(label, out_shape)
    return label

  def get_category_names(self, training):
    if not hasattr(self, '_category_names_cache'):
      ann_dict = self._train_annotations if training else self._val_annotations
      if 'categories' in ann_dict:
        categories = ann_dict['categories']
        self._category_names_cache = {c['id']: c for c in categories}
      else:
        self._category_names_cache = None
    return self._category_names_cache


@dataset_lib.DatasetRegistry.register('object_detection')
class CocoObjectDetectionDataset(CocoDataset):
  """Coco object detection dataset."""

  def extract(self, example, training):
    """Extracts needed features & annotations into a flat dictionary.

    Note:
      - label starts at 1 instead of 0, as 0 is reserved for special use
        (such as padding).
      - coordinates (e.g. bbox) are (normalized to be) in [0, 1].

    Args:
      example: `dict` of raw features.
      training: `bool` of training vs eval mode.

    Returns:
      example: `dict` of relevant features and labels.
    """
    features = {
        'image': tf.image.convert_image_dtype(example['image'], tf.float32),
        'image/id': example['image/id']
    }
    object_ids = example['objects']['id']
    bbox = example['objects']['bbox']
    iscrowd = example['objects']['is_crowd']

    # Drop crowded object annotation during both training and eval.
    not_is_crowd = tf.logical_not(iscrowd)
    object_ids = tf.boolean_mask(object_ids, not_is_crowd)
    bbox = tf.boolean_mask(bbox, not_is_crowd)
    iscrowd = tf.boolean_mask(iscrowd, not_is_crowd)

    labels = self._get_labels(object_ids, training)
    areas = self._get_areas(object_ids, training)
    labels = {
        'object/id': object_ids,
        'label': labels,
        'bbox': bbox,
        'area': areas,
        'is_crowd': iscrowd,
    }
    return features, labels


def load_annnotations(annotations_path):
  """Returns dict of object id to annotation."""
  logging.info('Loading annotations from %s', annotations_path)
  with tf.io.gfile.GFile(annotations_path, 'r') as f:
    annotations = json.load(f)
  return annotations


def xy2yx(seq):
  x = np.asarray(seq[::2]).reshape([-1, 1])
  y = np.asarray(seq[1::2]).reshape([-1, 1])
  return np.concatenate([y, x], axis=-1).reshape([-1]).tolist()
