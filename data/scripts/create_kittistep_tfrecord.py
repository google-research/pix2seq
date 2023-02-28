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
"""Generate tfrecord for kitti-step."""

import collections
import os
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf

flags.DEFINE_string('split', 'train', '')
flags.DEFINE_integer('shards', 25, '')
flags.DEFINE_integer('num_frames', 3, '')
flags.DEFINE_string('raw_image_dir', '', '')
flags.DEFINE_string('raw_ann_dir', '', '')
flags.DEFINE_string('output_dir', '', '')

FLAGS = flags.FLAGS

_INSTANCE_LABEL_DIVISOR = 1000
_ENCODED_INSTANCE_LABEL_DIVISOR = 256


def _decode_panoptic_map(panoptic_map_path: str) -> Optional[str]:
  """Decodes the panoptic map from encoded image file.


  Args:
    panoptic_map_path: Path to the panoptic map image file.

  Returns:
    Panoptic map as an encoded int32 numpy array bytes or None if not existing.
  """
  if not tf.io.gfile.exists(panoptic_map_path):
    return None
  with tf.io.gfile.GFile(panoptic_map_path, 'rb') as f:
    panoptic_map = np.array(Image.open(f)).astype(np.int32)
  semantic_map = panoptic_map[:, :, 0]
  instance_map = (
      panoptic_map[:, :, 1] * _ENCODED_INSTANCE_LABEL_DIVISOR +
      panoptic_map[:, :, 2])
  panoptic_map = semantic_map * _INSTANCE_LABEL_DIVISOR + instance_map
  return panoptic_map.tobytes()


def generate_tf_sequence_example(
    image_list, panoptic_map_list, filename, video_name):
  """Create tf sequence example."""
  assert len(image_list) == len(panoptic_map_list)
  height, width, c = image_list[0].shape
  assert c == 3
  frame_id = filename.split('.')[0]

  example_proto = tf.train.SequenceExample(
      context=tf.train.Features(
          feature={
              'image/filename':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[bytes(filename, 'utf-8')])),
              'image/format':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[bytes('png', 'utf-8')])),
              'image/channels':
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[c])),
              'image/height':
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[height])),
              'image/width':
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[width])),
              'image/segmentation/class/format':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[bytes('raw', 'utf-8')])),
              'video/sequence_id':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[bytes(video_name, 'utf-8')])),
              'video/frame_id':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[bytes(frame_id, 'utf-8')])),
          }),
      feature_lists=tf.train.FeatureLists(
          feature_list={
              'image/encoded_list':
                  tf.train.FeatureList(feature=[
                      tf.train.Feature(
                          bytes_list=tf.train.BytesList(
                              value=[tf.io.encode_png(image).numpy()]))
                      for image in image_list
                  ]),
              'image/segmentation/class/encoded_list':
                  tf.train.FeatureList(feature=[
                      tf.train.Feature(
                          bytes_list=tf.train.BytesList(value=[seg]))
                      for seg in panoptic_map_list
                  ]),
          }))

  return example_proto.SerializeToString()


def main(unused_argv):
  split = FLAGS.split
  raw_image_dir = FLAGS.raw_image_dir
  raw_ann_dir = FLAGS.raw_ann_dir
  num_frames = FLAGS.num_frames
  video_names = tf.io.gfile.listdir(os.path.join(raw_image_dir, split))
  if num_frames <= 0:
    assert FLAGS.shards <= 0
  shards = FLAGS.shards if FLAGS.shards > 0 else len(video_names)
  tf.io.gfile.makedirs(FLAGS.output_dir)
  writers = [
      tf.io.TFRecordWriter(os.path.join(
          FLAGS.output_dir,
          f'{split}_{num_frames}-{i:05d}-of-{shards:05d}.tfrecord'))
      for i in range(shards)
  ]

  k = 0
  for i, video_name in enumerate(video_names):
    frame_filenames = tf.io.gfile.listdir(
        os.path.join(raw_image_dir, split, video_name))

    # If larger than 3 frames, we only save one example per video, so only
    # getting the first n frames of that video.
    if num_frames > 3:
      frame_filenames = frame_filenames[:num_frames]

    all_images = collections.deque(
        maxlen=num_frames if num_frames > 0 else None)
    all_panoptic_maps = collections.deque(
        maxlen=num_frames if num_frames > 0 else None)
    for j, fn in enumerate(frame_filenames):
      logging.info('%s, %s', video_name, fn)

      # load the image.
      image = np.asarray(
          Image.open(
              tf.io.gfile.GFile(
                  os.path.join(raw_image_dir, split, video_name, fn), 'rb')))
      all_images.append(image)

      # load and decode the panoptic map.
      panoptic_map = _decode_panoptic_map(
          os.path.join(raw_ann_dir, split, video_name, fn))
      all_panoptic_maps.append(panoptic_map)

      if j >= num_frames - 1 and num_frames > 0:
        serialized_example = generate_tf_sequence_example(
            all_images, all_panoptic_maps, fn, video_name)
        writers[k % shards].write(serialized_example)
        k += 1

    # Write all frames out in the same example.
    if num_frames <= 0:
      serialized_example = generate_tf_sequence_example(
          all_images, all_panoptic_maps, frame_filenames[-1], video_name)
      writers[k % shards].write(serialized_example)
      k += 1

  for writer in writers:
    writer.close()


if __name__ == '__main__':
  app.run(main)
