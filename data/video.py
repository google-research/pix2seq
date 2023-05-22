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
"""Video datasets."""

import numpy as np
from data import dataset as dataset_lib
import tensorflow as tf


DAVIS_VIDEO_NAMES = [
    'bear',
    'bike-packing',
    'blackswan',
    'bmx-bumps',
    'bmx-trees',
    'boat',
    'boxing-fisheye',
    'breakdance',
    'breakdance-flare',
    'bus',
    'camel',
    'car-roundabout',
    'car-shadow',
    'car-turn',
    'cat-girl',
    'classic-car',
    'color-run',
    'cows',
    'crossing',
    'dance-jump',
    'dance-twirl',
    'dancing',
    'disc-jockey',
    'dog',
    'dog-agility',
    'dog-gooses',
    'dogs-jump',
    'dogs-scale',
    'drift-chicane',
    'drift-straight',
    'drift-turn',
    'drone',
    'elephant',
    'flamingo',
    'goat',
    'gold-fish',
    'hike',
    'hockey',
    'horsejump-high',
    'horsejump-low',
    'india',
    'judo',
    'kid-football',
    'kite-surf',
    'kite-walk',
    'koala',
    'lab-coat',
    'lady-running',
    'libby',
    'lindy-hop',
    'loading',
    'longboard',
    'lucia',
    'mallard-fly',
    'mallard-water',
    'mbike-trick',
    'miami-surf',
    'motocross-bumps',
    'motocross-jump',
    'motorbike',
    'night-race',
    'paragliding',
    'paragliding-launch',
    'parkour',
    'pigs',
    'planes-water',
    'rallye',
    'rhino',
    'rollerblade',
    'schoolgirls',
    'scooter-black',
    'scooter-board',
    'scooter-gray',
    'sheep',
    'shooting',
    'skate-park',
    'snowboard',
    'soapbox',
    'soccerball',
    'stroller',
    'stunt',
    'surf',
    'swing',
    'tennis',
    'tractor-sand',
    'train',
    'tuk-tuk',
    'upside-down',
    'varanus-cage',
    'walking']


def process_instance_id_map(instance_id_map, order, max_instances_per_image):
  if order == 'shuffle':
    num_instances = tf.reduce_max(instance_id_map)
    shuf_ids = tf.random.shuffle(tf.range(1, num_instances + 1))
    shuf_ids = tf.concat([[0], shuf_ids], -1)
    instance_id_map = tf.gather(shuf_ids, instance_id_map)
  elif order == 'random':
    rand_ids = tf.random.shuffle(tf.range(1, max_instances_per_image))
    rand_ids = tf.concat([[0], rand_ids], -1)
    instance_id_map = tf.gather(rand_ids, instance_id_map)
  else:
    assert order == 'none', 'Unknown order {}'.format(order)
  return instance_id_map


@dataset_lib.DatasetRegistry.register('davis_vps')
class DavisDataset(dataset_lib.TFRecordDataset):
  """The DAVIS 2017 video object segmentation dataset."""

  VIDEO_NAMES = DAVIS_VIDEO_NAMES

  def __init__(self, config):
    super().__init__(config)
    self._video_name_to_id_map = {
        v: i for i, v in enumerate(self.VIDEO_NAMES)}

  def get_feature_map(self, unused_training):
    """Returns feature map for parsing the TFExample."""
    context_features = {
        'image/format':
            tf.io.FixedLenFeature([], tf.string),
        'image/channels':
            tf.io.FixedLenFeature([], tf.int64),
        'image/height':
            tf.io.FixedLenFeature([], tf.int64),
        'image/width':
            tf.io.FixedLenFeature([], tf.int64),
        'video/frame_id':
            tf.io.FixedLenFeature([], tf.string),
        'video/name':
            tf.io.FixedLenFeature([], tf.string),
        'video/num_frames':
            tf.io.FixedLenFeature([], tf.int64),
    }
    sequence_features = {
        'video/frames':
            tf.io.FixedLenSequenceFeature([], tf.string),
        'video/segmentations':
            tf.io.FixedLenSequenceFeature([], tf.string),
    }
    return (context_features, sequence_features)

  def extract(self, example, training):
    """Extracts needed features & annotations into a flat dictionary.

    Args:
      example: `dict` of raw features.
      training: `bool` of training vs eval mode.

    Returns:
      example: `dict` of relevant features and labels.
    """
    def _get_video_id(video_name):
      video_name = video_name.decode('utf-8')
      video_id = self._video_name_to_id_map[
          video_name] if video_name in self._video_name_to_id_map else -1
      return np.asarray(video_id, dtype=np.int32)

    # Decode image and segmentation masks.
    frames = tf.map_fn(lambda x: tf.io.decode_png(x, channels=3),
                       example['video/frames'], tf.uint8)
    frames.set_shape([None, None, None, 3])
    segs = tf.map_fn(lambda x: tf.io.decode_png(x, channels=1),
                     example['video/segmentations'], tf.uint8)
    segs.set_shape([None, None, None, 1])

    new_example = {
        'video/frames': tf.image.convert_image_dtype(frames, tf.float32),
        'video/num_frames': tf.cast(example['video/num_frames'], tf.int32),
    }
    video_id = tf.numpy_function(
        _get_video_id, (example['video/name'],), (tf.int32,))
    video_id = tf.reshape(video_id, [])
    new_example['video/id'] = video_id

    segs = process_instance_id_map(
        tf.cast(segs, tf.int32), self.task_config.object_order,
        self.task_config.max_instances_per_image)
    new_example['label_map'] = tf.concat([tf.zeros_like(segs), segs], -1)
    return new_example


@dataset_lib.DatasetRegistry.register('kittistep_vps')
class KittiStepDataset(dataset_lib.TFRecordDataset):
  """The KITTI-STEP dataset."""

  def get_feature_map(self, training):
    """Returns feature map for parsing the TFExample."""
    del training
    context_features = {
        'image/filename':
            tf.io.FixedLenFeature([], tf.string),
        'image/format':
            tf.io.FixedLenFeature([], tf.string),
        'image/channels':
            tf.io.FixedLenFeature([], tf.int64),
        'image/height':
            tf.io.FixedLenFeature([], tf.int64),
        'image/width':
            tf.io.FixedLenFeature([], tf.int64),
        'image/segmentation/class/format':
            tf.io.FixedLenFeature([], tf.string),
        'video/sequence_id':
            tf.io.FixedLenFeature([], tf.string),
        'video/frame_id':
            tf.io.FixedLenFeature([], tf.string),
    }
    sequence_features = {
        'image/encoded_list':
            tf.io.FixedLenSequenceFeature([], tf.string),
        'image/segmentation/class/encoded_list':
            tf.io.FixedLenSequenceFeature([], tf.string),
    }
    return (context_features, sequence_features)

  def extract(self, example, training):
    """Extracts needed features & annotations into a flat dictionary.

    Args:
      example: `dict` of features.
      training: `bool` of training vs eval mode.

    Returns:
      example: `dict` of relevant features and labels.
    """
    def _get_video_id(video_name):
      video_name = video_name.decode('utf-8')
      video_id = int(video_name)
      return np.asarray(video_id, dtype=np.int32)

    def decode_label(parsed_label):
      flattened_label = tf.io.decode_raw(parsed_label, out_type=tf.int32)
      label_shape = tf.stack([
          example['image/height'], example['image/width'], 1
      ])
      label = tf.reshape(flattened_label, label_shape)
      return label

    decoded_frames = tf.map_fn(
        lambda x: tf.io.decode_png(x, channels=3),
        example['image/encoded_list'], tf.uint8)
    decoded_frames.set_shape([None, None, None, 3])
    video_id = tf.numpy_function(
        _get_video_id, (example['video/sequence_id'],), (tf.int32,))
    video_id = tf.reshape(video_id, [])
    new_example = {
        'video/frames': tf.image.convert_image_dtype(decoded_frames,
                                                     tf.float32),
        'video/num_frames': tf.shape(example['image/encoded_list'])[0],
        'video/id': video_id,
    }

    decoded_segs = tf.map_fn(
        decode_label, example['image/segmentation/class/encoded_list'],
        tf.int32)
    decoded_segs.set_shape([None, None, None, 1])

    semantic_label = tf.cast(
        decoded_segs // self.config.panoptic_label_divisor, tf.int32)
    # Replace void label with 0, and increment all class labels by 1.
    semantic_label = tf.where(semantic_label == self.config.ignore_label,
                              tf.zeros_like(semantic_label), semantic_label + 1)
    instance_label = tf.cast(
        decoded_segs % self.config.panoptic_label_divisor, tf.int32)
    if training:
      instance_label = process_instance_id_map(
          instance_label,
          self.task_config.object_order,
          self.task_config.max_instances_per_image)

    new_example['label_map'] = tf.concat([semantic_label, instance_label], -1)
    return new_example


@dataset_lib.DatasetRegistry.register('tfds_video')
class TFDSVideoDataset(dataset_lib.TFDSDataset):
  """Dataset for video classification datasets, e.g. UCF-101."""

  def extract(self, example, training):
    """Extracts needed features & annotations into a flat dictionary.

    Args:
      example: `dict` of raw features.
      training: `bool` of training vs eval mode.

    Returns:
      example: `dict` of relevant features and labels.
    """

    out = {}
    if 'rng' in example:
      out['rng'] = example.pop('rng')

    if 'image' in example:
      video = example['image'][None]
    elif isinstance(example['video'], dict) and 'frames' in example['video']:
      video = example['video']['frames']
    else:
      video = example['video']

    seq_len = self.config.get('seq_len', 4)
    video = video[:seq_len]

    # Convert video to float and normalize.
    assert video.dtype == tf.uint8
    video = tf.image.convert_image_dtype(video, tf.float32)
    out['video'] = video

    # Store original video shape (e.g. for correct evaluation metrics).
    out['shape'] = tf.shape(video)
    out['label'] = example['label']

    return out, dict(label=out['label'])

  @property
  def num_classes(self):
    return self.builder.info.features['label'].num_classes

  @property
  def num_train_examples(self):
    return sum([self.builder.info.splits[s].num_examples
                for s in self.config.train_split])

  @property
  def num_eval_examples(self):
    return sum([
        self.builder.info.splits[s].num_examples for s in self.config.eval_split
    ]) if not self.task_config.get('unbatch', False) else None
