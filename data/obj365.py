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
"""Object365 dataset class."""

import ml_collections
from data import dataset as dataset_lib
from data import decode_utils
import tensorflow as tf


@dataset_lib.DatasetRegistry.register('obj365')
class Obj365Dataset(dataset_lib.TFRecordDataset):
  """Dataset for Obj365 tasks."""

  def __init__(self, config: ml_collections.ConfigDict):
    """Constructs the dataset.

    Args:
      config: the model config.
    """
    super(dataset_lib.TFRecordDataset, self).__init__(config)

    if 'label_shift' in config.dataset:
      self.label_shift = config.dataset.label_shift
    else:
      self.label_shift = 0

  def _get_source_id(self, example):
    def _generate_source_id():
      return tf.strings.as_string(
          tf.strings.to_hash_bucket_fast(example['image/encoded'], 2**63 - 1))

    if self.config.get('regenerate_source_id', False):
      source_id = _generate_source_id()
    else:
      source_id = tf.cond(
          tf.greater(tf.strings.length(example['image/source_id']),
                     0), lambda: example['image/source_id'],
          _generate_source_id)
    return source_id

  def _decode_masks(self, example):
    """Decode a set of PNG masks to the tf.float32 tensors."""
    def _decode_png_mask(png_bytes):
      mask = tf.squeeze(
          tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8), axis=-1)
      mask = tf.cast(mask, dtype=tf.float32)
      mask.set_shape([None, None])
      return mask

    height = example['image/height']
    width = example['image/width']
    masks = example['image/object/mask']
    return tf.cond(
        tf.greater(tf.size(masks), 0),
        lambda: tf.map_fn(_decode_png_mask, masks, dtype=tf.float32),
        lambda: tf.zeros([0, height, width], dtype=tf.float32))

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
        'image': decode_utils.decode_image(example),
        'image/id': self._get_source_id(example),
    }
    bbox = decode_utils.decode_boxes(example)
    labels = {
        'bbox_orig': bbox,
        'bbox': bbox,
        'is_crowd': decode_utils.decode_is_crowd(example),
        'label': example['image/object/class/label'] + self.label_shift,
        'area': decode_utils.decode_areas(example),
    }
    return features, labels

  @property
  def num_train_examples(self):
    return {
        'obj365': 1662289,
        'obj365v1': 608606,
        'obj365v2': 1662289
    }[self.config.dataset_name]

  @property
  def num_eval_examples(self):
    return {
        'obj365': 80000,
        'obj365v1': 30000,
        'obj365v2': 80000
    }[self.config.dataset_name]

  @property
  def num_classes(self):
    return 365

  def get_feature_map(self):
    """Returns feature map for parsing the TFExample."""
    image_feature_map = decode_utils.get_feature_map_for_image()
    detection_feature_map = decode_utils.get_feature_map_for_object_detection()
    feature_map = {**image_feature_map, **detection_feature_map}
    if self.config.get('include_mask', False):
      feature_map.update({'image/object/mask': tf.io.VarLenFeature(tf.string)})
    return feature_map
