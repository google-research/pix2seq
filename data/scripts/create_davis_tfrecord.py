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
"""Generate tfrecord for DAVIS2017."""

import collections
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf

flags.DEFINE_string('split', 'train', 'train or val')
flags.DEFINE_integer('shards', 25, '')
flags.DEFINE_integer('num_frames', 3, '')
flags.DEFINE_string('data_dir', '', '')
flags.DEFINE_string('output_dir', '', '')

FLAGS = flags.FLAGS


# There is one video ('tennis' in train) that labels one object as 255.
def handle_mislabel_255(arr):
  maxm = np.max(arr[arr != 255])
  arr[arr == 255] = maxm + 1
  return arr


def generate_tf_sequence_example(
    image_list, segmentation_list, filename, video_name):
  """Create tf sequence example."""
  num_frames = len(image_list)
  assert len(image_list) == len(segmentation_list)
  height, width, c = image_list[0].shape
  assert c == 3
  frame_id = filename.split('.')[0]

  example_proto = tf.train.SequenceExample(
      context=tf.train.Features(
          feature={
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
              'video/name':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[bytes(video_name, 'utf-8')])),
              'video/frame_id':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[bytes(frame_id, 'utf-8')])),
              'video/num_frames':
                  tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[num_frames])),
          }),
      feature_lists=tf.train.FeatureLists(
          feature_list={
              'video/frames':
                  tf.train.FeatureList(feature=[
                      tf.train.Feature(
                          bytes_list=tf.train.BytesList(
                              value=[tf.io.encode_png(image).numpy()]))
                      for image in image_list
                  ]),
              'video/segmentations':
                  tf.train.FeatureList(feature=[
                      tf.train.Feature(
                          bytes_list=tf.train.BytesList(value=[
                              tf.io.encode_png(tf.expand_dims(seg, -1)).numpy()
                          ])) for seg in segmentation_list
                  ]),
          }))

  return example_proto.SerializeToString()


def main(unused_argv):
  split = FLAGS.split
  data_dir = FLAGS.data_dir
  images_dir = os.path.join(data_dir, 'JPEGImages/480p')
  annotation_dir = os.path.join(data_dir, 'Annotations_unsupervised/480p/')
  video_names = [
      s.strip() for s in tf.io.gfile.GFile(
          os.path.join(data_dir, f'ImageSets/2017/{split}.txt')).readlines()]

  num_frames = FLAGS.num_frames
  shards = FLAGS.shards
  output_dir = FLAGS.output_dir
  writers = [
      tf.io.TFRecordWriter(os.path.join(
          output_dir,
          f'{split}_{num_frames}-{i:05d}-of-{shards:05d}.tfrecord'))
      for i in range(shards)
  ]

  k = 0
  for i, video_name in enumerate(video_names):
    image_filenames = tf.io.gfile.listdir(
        os.path.join(images_dir, video_name))
    ann_filenames = tf.io.gfile.listdir(
        os.path.join(annotation_dir, video_name))

    all_images = collections.deque(
        maxlen=num_frames if num_frames > 0 else None)
    all_segs = collections.deque(
        maxlen=num_frames if num_frames > 0 else None)
    for j, (image_f, ann_f) in enumerate(zip(image_filenames, ann_filenames)):
      logging.info('%s, %s', video_name, image_f)

      # load the image.
      image = np.asarray(
          Image.open(
              tf.io.gfile.GFile(
                  os.path.join(images_dir, video_name, image_f), 'rb')))
      all_images.append(image)

      # load the segmentations.
      data = np.array(
          Image.open(
              tf.io.gfile.GFile(
                  os.path.join(annotation_dir, video_name, ann_f), 'rb')))
      data = handle_mislabel_255(data)
      all_segs.append(data)

      if j >= num_frames - 1 and num_frames > 0:
        serialized_example = generate_tf_sequence_example(
            all_images, all_segs, image_f, video_name)
        writers[k % shards].write(serialized_example)
        k += 1

    # Write all frames out in the same example.
    if num_frames <= 0:
      serialized_example = generate_tf_sequence_example(
          all_images, all_segs, image_filenames[-1], video_name)
      writers[k % shards].write(serialized_example)
      k += 1

  for writer in writers:
    writer.close()


if __name__ == '__main__':
  app.run(main)
