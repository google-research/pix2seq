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
"""Merge COCO annotation json file with tfrecord files.

This is used to merge the generated object detection annotations with the
groundtruth tfrecord files, to generate new tfrecord files that can be used
for downstream task inference such as instance segmentation and keypoint
detection.

When merging, the image features (image/encoded, image/source_id, etc) are kept
the same. Object features related to detection (bbox, is_crowd, label) are
populated from the json annotation file. Downstream task features (
segmentation, keypoint) are padded.
"""

import collections
import json
import os

from absl import app
from absl import flags
from absl import logging
from data.scripts import tfrecord_lib
import tensorflow as tf

flags.DEFINE_string('tfrecord_path', '', 'Tfrecord file pattern.')
flags.DEFINE_string('annotation_path', '', 'JSON annotation file path.')
flags.DEFINE_string('output_dir', None, 'Output directory')

FLAGS = flags.FLAGS


COPY_FEATURE_LIST = ['image/encoded', 'image/source_id', 'image/format',
                     'image/filename', 'image/height', 'image/width',
                     'image/key/sha256', 'image/caption']


def load_instance_annotations(annotation_path):
  """Load instance annotations.

  Args:
    annotation_path: str. Path to the annotation file.

  Returns:
    category_id_to_name_map: dict of category ids to category names.
    img_to_ann: a dict of image_id to annotation.
  """
  with tf.io.gfile.GFile(annotation_path, 'r') as f:
    annotations = json.load(f)

  img_to_ann = collections.defaultdict(list)
  for ann in annotations['annotations']:
    image_id = ann['image_id']
    img_to_ann[image_id].append(ann)

  category_id_to_name_map = dict(
      (element['id'], element['name']) for element in annotations['categories'])

  return category_id_to_name_map, img_to_ann


def coco_annotations_to_lists(obj_annotations, id_to_name_map):
  """Converts COCO annotations to feature lists.

  Args:
    obj_annotations: a list of object annotations.
    id_to_name_map: category id to category name map.

  Returns:
    a dict of list features.
  """

  data = dict((k, list()) for k in [
      'xmin', 'xmax', 'ymin', 'ymax', 'is_crowd', 'category_id',
      'category_names', 'area', 'score'])

  for ann in obj_annotations:
    (x, y, width, height) = tuple(ann['bbox'])
    if width > 0. and height > 0.:  # Only keep valid boxes.
      data['xmin'].append(float(x))
      data['xmax'].append(float(x + width))
      data['ymin'].append(float(y))
      data['ymax'].append(float(y + height))
      data['is_crowd'].append(ann['iscrowd'])
      category_id = int(ann['category_id'])
      data['category_id'].append(category_id)
      data['category_names'].append(id_to_name_map[category_id].encode('utf8'))
      data['area'].append(float(height * width))
      data['score'].append(ann['score'])

  return data


def obj_annotations_to_feature_dict(obj_annotations, id_to_name_map):
  """Convert COCO annotations to an encoded feature dict.

  Args:
    obj_annotations: a list of object annotations.
    id_to_name_map: category id to category name map.

  Returns:
    a dict of tf features, and the number of instances.
  """

  data = coco_annotations_to_lists(obj_annotations, id_to_name_map)
  feature_dict = {
      'image/object/bbox/xmin':
          tfrecord_lib.convert_to_feature(
              data['xmin'], value_type='float_list'),
      'image/object/bbox/xmax':
          tfrecord_lib.convert_to_feature(
              data['xmax'], value_type='float_list'),
      'image/object/bbox/ymin':
          tfrecord_lib.convert_to_feature(
              data['ymin'], value_type='float_list'),
      'image/object/bbox/ymax':
          tfrecord_lib.convert_to_feature(
              data['ymax'], value_type='float_list'),
      'image/object/class/text':
          tfrecord_lib.convert_to_feature(
              data['category_names'], value_type='bytes_list'),
      'image/object/class/label':
          tfrecord_lib.convert_to_feature(
              data['category_id'], value_type='int64_list'),
      'image/object/is_crowd':
          tfrecord_lib.convert_to_feature(
              data['is_crowd'], value_type='int64_list'),
      'image/object/area':
          tfrecord_lib.convert_to_feature(
              data['area'], value_type='float_list'),
      'image/object/score':
          tfrecord_lib.convert_to_feature(
              data['score'], value_type='float_list'),
  }
  return feature_dict, len(data['xmin'])


def update_tfrecord_file(tfrecord_path, image_to_anns, category_id_to_name_map,
                         output_path):
  """Merge one tfrecord file with annotations.

  Args:
    tfrecord_path: string, the input tfrecord path.
    image_to_anns: a dict of image_id to annotation.
    category_id_to_name_map: dict of category ids to category names.
    output_path: string, the output tfrecord file path.
  """
  dataset = tf.data.TFRecordDataset(tfrecord_path)

  with tf.io.TFRecordWriter(output_path) as writer:
    for serialized_ex in dataset:
      ex = tf.train.Example()
      ex.ParseFromString(serialized_ex.numpy())

      image_id = int(ex.features.feature['image/source_id'].bytes_list.value[0])
      anns = image_to_anns[image_id]

      # Copy the following features from current tf example.
      feature_dict = {}
      for f in COPY_FEATURE_LIST:
        feature_dict[f] = ex.features.feature[f]

      # Populate the object detection features from json annotations.
      det_features, num_bbox = obj_annotations_to_feature_dict(
          anns, category_id_to_name_map)
      feature_dict.update(det_features)

      # Pad the segmentation and keypoint features.
      feature_dict.update({
          'image/object/segmentation_v':
              tfrecord_lib.convert_to_feature([], value_type='float_list'),
          'image/object/segmentation_sep':
              tfrecord_lib.convert_to_feature(
                  [0] * (num_bbox + 1), value_type='int64_list'),
          'image/object/keypoints_v':
              tfrecord_lib.convert_to_feature([], value_type='float_list'),
          'image/object/keypoints_sep':
              tfrecord_lib.convert_to_feature(
                  [0] * (num_bbox + 1), value_type='int64_list'),
          'image/object/num_keypoints':
              tfrecord_lib.convert_to_feature(
                  [0] * num_bbox, value_type='int64_list'),
      })

      new_ex = tf.train.Example(
          features=tf.train.Features(feature=feature_dict)).SerializeToString()
      writer.write(new_ex)


def main(unused_argv):
  category_id_to_name_map, image_to_anns = load_instance_annotations(
      FLAGS.annotation_path)

  tfrecord_lib.check_and_make_dir(FLAGS.output_dir)

  tfrecord_paths = tf.io.gfile.glob(FLAGS.tfrecord_path)
  for tfrecord_path in tfrecord_paths:
    output_path = os.path.join(FLAGS.output_dir,
                               os.path.basename(tfrecord_path))
    update_tfrecord_file(tfrecord_path, image_to_anns, category_id_to_name_map,
                         output_path)
    logging.info('Finished writing file %s', output_path)


if __name__ == '__main__':
  app.run(main)
