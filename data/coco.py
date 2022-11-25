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

import numpy as np
import utils
import vocab
from data import dataset as dataset_lib
from data import decode_utils
import tensorflow as tf


def _xy_to_yx(tensor):
  """Convert a tensor in xy order into yx order.

  Args:
    tensor: of shape [num_instances, points * 2] in the xy order.

  Returns:
    a tensor of shape [num_instances, points, 2] in the yx order.
  """
  max_points = tf.shape(tensor)[1] / 2
  t = tf.reshape(tensor, [-1, max_points, 2])
  t = tf.stack([t[:, :, 1], t[:, :, 0]], axis=2)
  return tf.reshape(t, [-1, max_points * 2])


@dataset_lib.DatasetRegistry.register('coco/2017_object_detection')
class CocoObjectDetectionTFRecordDataset(dataset_lib.TFRecordDataset):
  """Coco object detection dataset."""

  def get_feature_map(self):
    """Returns feature map for parsing the TFExample."""
    image_feature_map = decode_utils.get_feature_map_for_image()
    detection_feature_map = decode_utils.get_feature_map_for_object_detection()
    return {**image_feature_map, **detection_feature_map}

  def filter_example(self, example, training):
    # Filter out examples with no instances.
    if training:
      return tf.shape(example['image/object/bbox/xmin'])[0] > 0
    else:
      return True

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
        'image/id': tf.strings.to_number(example['image/source_id'], tf.int64),
    }

    bbox = decode_utils.decode_boxes(example)
    scale = 1. / utils.tf_float32(tf.shape(features['image'])[:2])
    bbox = utils.scale_points(bbox, scale)

    labels = {
        'label': example['image/object/class/label'],
        'bbox': bbox,
        'area': decode_utils.decode_areas(example),
        'is_crowd': decode_utils.decode_is_crowd(example),
    }

    return features, labels


@dataset_lib.DatasetRegistry.register('coco/2017_instance_segmentation')
class CocoInstanceSegmentationTFRecordDataset(dataset_lib.TFRecordDataset):
  """Coco instance segmentation dataset."""

  def get_feature_map(self):
    """Returns feature map for parsing the TFExample."""
    image_feature_map = decode_utils.get_feature_map_for_image()
    detection_feature_map = decode_utils.get_feature_map_for_object_detection()
    seg_feature_map = decode_utils.get_feature_map_for_instance_segmentation()
    return {**image_feature_map, **detection_feature_map, **seg_feature_map}

  def filter_example(self, example, training):
    # Filter out examples with no instances, to avoid error when converting
    # RaggedTensor to tensor: `Invalid first partition input. Tensor requires
    # at least one element.`
    return tf.shape(example['image/object/bbox/xmin'])[0] > 0

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
    assert not self.task_config.shuffle_polygon_start_point
    features = {
        'image': decode_utils.decode_image(example),
        'image/id': tf.strings.to_number(example['image/source_id'], tf.int64),
    }

    max_points = self.task_config.max_points_per_object
    polygons = example['image/object/segmentation'].to_tensor(
        default_value=vocab.PADDING_FLOAT,
        shape=[None, max_points * 2])
    polygons = _xy_to_yx(polygons)

    bbox = decode_utils.decode_boxes(example)
    labels = example['image/object/class/label']
    iscrowd = decode_utils.decode_is_crowd(example)
    areas = decode_utils.decode_areas(example)
    scores = decode_utils.decode_scores(example)

    # Drop crowded object annotation during both training and eval.
    is_valid = tf.logical_not(iscrowd)
    bbox = tf.boolean_mask(bbox, is_valid)
    iscrowd = tf.boolean_mask(iscrowd, is_valid)
    labels = tf.boolean_mask(labels, is_valid)
    areas = tf.boolean_mask(areas, is_valid)
    polygons = tf.boolean_mask(polygons, is_valid)
    scores = tf.boolean_mask(scores, is_valid)

    scale = 1. / utils.tf_float32(tf.shape(features['image'])[:2])
    labels = {
        'label': labels,
        'bbox': utils.scale_points(bbox, scale),
        'area': areas,
        'is_crowd': iscrowd,
        'polygon': utils.scale_points(polygons, scale),
        'scores': scores
    }

    return features, labels


@dataset_lib.DatasetRegistry.register('coco/2017_keypoint_detection')
class CocoKeypointDetectionTFRecordDataset(dataset_lib.TFRecordDataset):
  """Coco keypoint detection dataset."""

  def get_feature_map(self):
    """Returns feature map for parsing the TFExample."""
    image_feature_map = decode_utils.get_feature_map_for_image()
    detection_feature_map = decode_utils.get_feature_map_for_object_detection()
    key_feature_map = decode_utils.get_feature_map_for_keypoint_detection()
    return {**image_feature_map, **detection_feature_map, **key_feature_map}

  def filter_example(self, example, training):
    # Filter out examples without keypoints.
    if training:
      return tf.reduce_sum(example['image/object/num_keypoints']) > 0
    else:
      return tf.shape(example['image/object/bbox/xmin'])[0] > 0

  def set_invisible_points(self, keypoints):
    segs = []
    num_points = np.shape(keypoints)[1] // 3
    for seg in keypoints:
      out = []
      for i in range(num_points):
        if seg[i * 3 + 2] == 0:  # Not labeled
          seg[i * 3] = seg[i * 3 + 1] = vocab.INVISIBLE_FLOAT
        out.extend([seg[i * 3], seg[i * 3 + 1]])  # Drop visibility flags.
      segs.append(out)
    return np.asarray(segs, dtype=np.float32)

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
        'image/id': tf.strings.to_number(example['image/source_id'], tf.int64),
    }

    max_points = self.task_config.max_points_per_object
    keypoints = example['image/object/keypoints'].to_tensor(
        default_value=vocab.PADDING_FLOAT, shape=[None, max_points * 3])
    keypoints = tf.numpy_function(self.set_invisible_points, (keypoints,),
                                  tf.float32)
    keypoints.set_shape([None, max_points * 2])
    keypoints = _xy_to_yx(keypoints)

    bbox = decode_utils.decode_boxes(example)
    num_keypoints = example['image/object/num_keypoints']
    labels = example['image/object/class/label']
    iscrowd = decode_utils.decode_is_crowd(example)
    areas = decode_utils.decode_areas(example)
    scores = decode_utils.decode_scores(example)

    # Only keep non-crowded person objects during both training and eval.
    is_valid = tf.logical_and(
        tf.logical_and(tf.equal(labels, 1), tf.logical_not(iscrowd)),
        tf.math.greater_equal(scores, self.task_config.min_bbox_score))

    if training:  # Drop person without any anno keypoints during training.
      has_keypoints = tf.greater(num_keypoints, 0)
      is_valid = tf.logical_and(is_valid, has_keypoints)

    bbox = tf.boolean_mask(bbox, is_valid)
    iscrowd = tf.boolean_mask(iscrowd, is_valid)
    labels = tf.boolean_mask(labels, is_valid)
    areas = tf.boolean_mask(areas, is_valid)
    keypoints = tf.boolean_mask(keypoints, is_valid)
    num_keypoints = tf.boolean_mask(num_keypoints, is_valid)
    scores = tf.boolean_mask(scores, is_valid)

    scale = 1. / utils.tf_float32(tf.shape(features['image'])[:2])
    labels = {
        'label': labels,
        'bbox': utils.scale_points(bbox, scale),
        'area': areas,
        'is_crowd': iscrowd,
        'keypoints': utils.scale_points(keypoints, scale),
        'num_keypoints': num_keypoints,
        'scores': scores
    }

    return features, labels


@dataset_lib.DatasetRegistry.register('coco/2017_captioning')
class CocoCaptioningTFRecordDataset(dataset_lib.TFRecordDataset):
  """Coco captioning dataset."""

  def get_feature_map(self):
    """Returns feature map for parsing the TFExample."""
    image_feature_map = decode_utils.get_feature_map_for_image()
    cap_feature_map = decode_utils.get_feature_map_for_captioning()
    return {**image_feature_map, **cap_feature_map}

  def filter_example(self, example, training):
    # Filter out examples without captions.
    if training:
      return tf.shape(example['image/caption'])[0] > 0
    else:
      return True

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
        'image/id': tf.strings.to_number(example['image/source_id'], tf.int64),
    }

    labels = {
        'captions':
            example['image/caption'][:self.task_config.captions_per_image],
    }

    return features, labels
