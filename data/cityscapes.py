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
"""Cityscapes dataset."""
from data import dataset as dataset_lib
import tensorflow as tf


@dataset_lib.DatasetRegistry.register('cityscapes_panoptic')
class CityscapesPanopticDataset(dataset_lib.TFRecordDataset):
  """Cityscapes panoptic dataset."""

  def get_feature_map(self, training):
    """Returns feature map for parsing the TFExample."""
    del training
    return {
        'image/encoded':
            tf.io.FixedLenFeature([], tf.string),
        'image/segmentation/class/encoded':
            tf.io.FixedLenFeature([], tf.string),
        'image/height':
            tf.io.FixedLenFeature([], tf.int64),
        'image/width':
            tf.io.FixedLenFeature([], tf.int64),
        'image/filename':
            tf.io.FixedLenFeature([], tf.string),
    }

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
    # Decode image and label.
    image = tf.io.decode_image(example['image/encoded'], channels=3)
    image.set_shape([1024, 2048, 3])
    label = example['image/segmentation/class/encoded']
    label = tf.io.decode_raw(
        example['image/segmentation/class/encoded'], out_type=tf.int32)
    label_shape = tf.stack([1024, 2048])
    label = tf.reshape(label, label_shape)

    # Map instance ids to range(1, num_instance + 1)
    unique_instance_ids, _ = tf.unique(tf.reshape(label, [-1]))
    num_instances = tf.size(unique_instance_ids)
    new_instance_ids = tf.random.shuffle(tf.range(1, num_instances + 1))
    def map_ids(x, src_ids, tgt_ids):
      """Convert object ids into semantic classes."""
      x = tf.equal(x[:, :, tf.newaxis], src_ids[tf.newaxis, tf.newaxis, :])
      x = tf.reduce_sum(tf.cast(x, tgt_ids.dtype) *
                        tgt_ids[tf.newaxis, tf.newaxis, :], -1)
      return x
    identity = map_ids(label, unique_instance_ids, new_instance_ids)

    # label = class * max_instances_per_class + per_class_instance_id
    semantic = label // self.config.max_instances_per_class

    ignore_mask = tf.logical_not(tf.logical_and(
        tf.greater_equal(semantic, 0),
        tf.less(semantic, self.config.num_classes -
                1)))  # num_classes includes padding class.
    # 0 is reserved for background and labels which are to be ignored.
    semantic = tf.where(ignore_mask, tf.zeros_like(semantic), semantic + 1)
    identity = tf.where(ignore_mask, tf.zeros_like(identity), identity)

    return {
        'image':
            tf.image.convert_image_dtype(image, tf.float32),
        # TODO(srbs): Find another hashing strategy that does not have
        # collisions possibly by leveraging the structure of the filename which
        # is <city>_123456_123456.
        # Coco metrics would fail if there are duplicate image ids in preds or
        # gt.
        'image/id':
            tf.strings.to_hash_bucket(
                example['image/filename'], num_buckets=1000000000),
        'label_map': tf.stack([semantic, identity], -1)
    }
