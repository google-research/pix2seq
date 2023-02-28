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
"""Transform classes."""

import abc
import copy
import numpy as np

import ml_collections
import registry
import utils
from data import data_utils
import tensorflow as tf

TransformRegistry = registry.Registry()


DEFAULT_IMAGE_KEY = 'image'
DEFAULT_ORIG_IMAGE_SIZE_KEY = 'orig_image_size'
DEFAULT_BBOX_KEY = 'bbox'


class Transform(object):
  """Transform class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.config = config

  @abc.abstractmethod
  def process_example(self, example: dict[str, tf.Tensor]):
    """Implement this function to perform example-level processing."""


@TransformRegistry.register('copy_fields')
class CopyFields(Transform):
  """Copy features to new feature names.

  Required fields in config:
    inputs: names of the features to be copied.
    outputs: names of the new features.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    for i, o in zip(self.config.inputs, self.config.outputs):
      example[o] = copy.copy(example[i])
    return example


@TransformRegistry.register('record_original_image_size')
class RecordOriginalImageSize(Transform):
  """Record the original image size.

  Required fields in config:
    image_key: optional name of image feature. Defaults to 'image'.
    original_image_size_key: optional name of the original image size feature.
      Defaults to 'orig_image_size'.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    image_key = self.config.get('image_key', 'image')
    orig_image_size_key = self.config.get('original_image_size_key',
                                          DEFAULT_ORIG_IMAGE_SIZE_KEY)
    example[orig_image_size_key] = tf.shape(example[image_key])[:2]
    return example


@TransformRegistry.register('convert_image_dtype_float32')
class ConvertImageDtypeFloat32(Transform):
  """Convert image dtype to float32.

  Required fields in config:
    inputs: names of applicable fields in the example.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    for k in self.config.inputs:
      if example[k].dtype != tf.float32:
        example[k] = tf.image.convert_image_dtype(example[k], tf.float32)
    return example


@TransformRegistry.register('resize_image')
class ResizeImage(Transform):
  """Resize image.

  Required fields in config:
    inputs: names of applicable fields in the example.
    target_size: (height, width) tuple.
    resize_method: Optional[List[str]]. Defaults to bilinear.
    antialias: Optional[List[bool]]. Defaults to False.
    preserve_aspect_ratio: Optional[List[bool]]. Defaults to True.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    num_inputs = len(self.config.inputs)
    resize_methods = self.config.get('resize_method', ['bilinear'] * num_inputs)
    antialias_list = self.config.get('antialias', [False] * num_inputs)
    preserve_ar = self.config.get('preserve_aspect_ratio', [True] * num_inputs)

    for k, resize_method, antialias, p_ar in zip(
        self.config.inputs, resize_methods, antialias_list, preserve_ar):
      example[k] = tf.image.resize(
          example[k], self.config.target_size, method=resize_method,
          antialias=antialias, preserve_aspect_ratio=p_ar)
    return example


@TransformRegistry.register('scale_jitter')
class ScaleJitter(Transform):
  """Scale jittering.

  Required fields in config:
    inputs: names of applicable fields in the example.
    min_scale: float.
    max_scale: float.
    target_size: (height, width) tuple.
    resize_method: Optional[List[str]]. Defaults to bilinear.
    antialias: Optional[List[bool]]. Defaults to False.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    target_height, target_width = self.config.target_size
    min_scale, max_scale = self.config.min_scale, self.config.max_scale

    k = self.config.inputs[0]
    input_size = tf.cast(tf.shape(example[self.config.inputs[0]])[:2],
                         tf.float32)
    output_size = tf.constant([target_height, target_width], tf.float32)
    random_scale = tf.random.uniform([], min_scale, max_scale)
    random_scale_size = tf.multiply(output_size, random_scale)
    scale = tf.minimum(
        random_scale_size[0] / input_size[0],
        random_scale_size[1] / input_size[1]
    )
    scaled_size = tf.cast(tf.multiply(input_size, scale), tf.int32)

    num_inputs = len(self.config.inputs)
    resize_methods = self.config.get('resize_method', ['bilinear'] * num_inputs)
    antialias_list = self.config.get('antialias', [False] * num_inputs)
    for k, resize_method, antialias in zip(self.config.inputs,
                                           resize_methods, antialias_list):
      example[k] = tf.image.resize(
          example[k], tf.cast(scaled_size, tf.int32),
          method=resize_method, antialias=antialias)
    return example


@TransformRegistry.register('fixed_size_crop')
class FixedSizeCrop(Transform):
  """Fixed size crop.

  Fields in config:
    inputs: names of applicable fields in the example.
    target_size: (height, width) tuple.
    object_coordinate_keys: optional list of strings. Keys for coordinate
      features that describe objects, e.g. 'bbox' for object detection.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    target_height, target_width = self.config.target_size
    input_size = tf.shape(example[self.config.inputs[0]])[:2]
    output_size = tf.stack([target_height, target_width])

    max_offset = tf.subtract(input_size, output_size)
    max_offset = tf.cast(tf.maximum(max_offset, 0), tf.float32)
    offset = tf.multiply(max_offset, tf.random.uniform([], 0.0, 1.0))
    offset = tf.cast(offset, tf.int32)

    region = (offset[0], offset[1],
              tf.minimum(output_size[0], input_size[0] - offset[0]),
              tf.minimum(output_size[1], input_size[1] - offset[1]))
    object_coordinate_keys = self.config.get('object_coordinate_keys', [])

    return data_utils.crop(example, region, self.config.inputs,
                           object_coordinate_keys)


@TransformRegistry.register('random_horizontal_flip')
class RandomHorizontalFlip(Transform):
  """Random horizontal flip.

  Required fields in config:
    inputs: names of applicable fields in the example.
    bbox_keys: optional. Names of bbox fields.
    keypoints_keys: optional. Names of bbox fields.
    polygon_keys: optional. Names of polygon fields.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    inputs = {k: example[k] for k in self.config.inputs}
    boxes = {k: example[k] for k in self.config.get('bbox_keys', [])}
    keypoints = {k: example[k] for k in self.config.get('keypoints_keys', [])}
    polygons = {k: example[k] for k in self.config.get('polygon_keys', [])}

    with tf.name_scope('RandomHorizontalFlip'):
      coin_flip = tf.random.uniform([]) > 0.5
      if coin_flip:
        inputs = {k: tf.image.flip_left_right(v) for k, v in inputs.items()}
        boxes = {k: data_utils.flip_boxes_left_right(v)
                 for k, v in boxes.items()}
        keypoints = {k: data_utils.flip_keypoints_left_right(v)
                     for k, v in keypoints.items()}
        polygons = {k: data_utils.flip_polygons_left_right(v)
                    for k, v in polygons.items()}

    example.update(inputs)
    example.update(boxes)
    example.update(keypoints)
    example.update(polygons)
    return example


@TransformRegistry.register('random_color_jitter')
class RandomColorJitter(Transform):
  """Random color jittering.

  Fields in config:
    inputs: names of applicable fields in the example.
    color_jitter_strength: optional float.
    clip_by_value: optional tuple of float (min, max). Default is (0., 1.).
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    if self.config.get('color_jitter_strength', 0) > 0:
      clip_min, clip_max = self.config.get('clip_by_value', (0., 1.))
      for k in self.config.inputs:
        example[k] = data_utils.random_color_jitter(
            example[k], strength=self.config.color_jitter_strength,
            impl='simclrv2')
        example[k] = tf.clip_by_value(example[k], clip_min, clip_max)
    return example


@TransformRegistry.register('filter_invalid_objects')
class FilterInvalidObjects(Transform):
  """Filter objects with invalid bboxes.

  Required fields in config:
    inputs: names of applicable fields in the example.
    bbox_key: optional name of the bbox field. Defaults to 'bbox'.
    filter_keys: optional. Names of fields that, if True, the object will be
      filtered out. E.g. 'is_crowd', the objects with 'is_crowd=True' will be
      filtered out.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    bbox = example[self.config.get('bbox_key', DEFAULT_BBOX_KEY)]
    box_valid = tf.logical_and(bbox[:, 2] > bbox[:, 0], bbox[:, 3] > bbox[:, 1])
    for k in self.config.get('filter_keys', []):
      box_valid = tf.logical_and(box_valid, tf.logical_not(example[k]))
    valid_indices = tf.where(box_valid)[:, 0]
    for k in self.config.inputs:
      example[k] = tf.gather(example[k], valid_indices)
    return example


@TransformRegistry.register('reorder_object_instances')
class ReorderObjectInstances(Transform):
  """Reorder object instances, Must be _before_ padding to max instances.

  Required fields in config:
    inputs: names of applicable fields in the example.
    order: order to reorder object instances, in ['random', 'area', 'dist2ori',
      'scores'].
    bbox_key: optional name of the bbox field. Defaults to 'bbox'. This is only
      used for 'area' and 'dist2ori' methods.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    if self.config.get('bbox_key', DEFAULT_BBOX_KEY) in example:
      bbox = example[self.config.get('bbox_key', DEFAULT_BBOX_KEY)]
      assert bbox.shape.rank == 2, 'Must be unbatched'
      bbox = tf.reshape(bbox, [-1, 2, 2])
    else:
      bbox = None

    order = self.config.order
    if order == 'none':
      return example
    elif order == 'random':
      if bbox is not None:
        num_instances = tf.shape(bbox)[0]
      else:
        num_instances = tf.shape(example[self.config.inputs[0]])[0]
      idx = tf.random.shuffle(tf.range(num_instances))
    elif order == 'area':
      areas = tf.cast(tf.reduce_prod(bbox[:, 1, :] - bbox[:, 0, :], axis=1),
                      tf.int64)  # approximated size.
      idx = tf.argsort(areas, direction='DESCENDING')
    elif order == 'dist2ori':
      y, x = bbox[:, 0], bbox[:, 1]  # using top-left corner.
      dist2ori = tf.square(y) + tf.square(x)
      idx = tf.argsort(dist2ori, direction='ASCENDING')
    elif order == 'scores':
      idx = tf.argsort(example['scores'], direction='DESCENDING')
    else:
      raise ValueError('Unknown order {}'.format(order))

    for k in self.config.inputs:
      example[k] = tf.gather(example[k], idx)
    return example


@TransformRegistry.register('inject_noise_bbox')
class InjectNoiseBbox(Transform):
  """Inject noise bbox.

  Required fields in config:
    max_instances_per_image: int.
    bbox_key: optional name of the bbox field. Defaults to 'bbox'.
    bbox_label_key: optional name of the bbox label field. Defaults to 'label'.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    bbox_key = self.config.get('bbox_key', DEFAULT_BBOX_KEY)
    bbox_label_key = self.config.get('bbox_label_key', 'label')

    num_instances = tf.shape(example[bbox_key])[0]
    if num_instances < self.config.max_instances_per_image:
      n_noise_bbox = self.config.max_instances_per_image - num_instances
      example[bbox_key], example[bbox_label_key] = data_utils.augment_bbox(
          example[bbox_key], example[bbox_label_key], 0., n_noise_bbox)
    return example


@TransformRegistry.register('pad_image_to_max_size')
class PadImageToMaxSize(Transform):
  """Pad image to target size (height, width, 3).

  - Replace input fields with padded images.
  - Re-scale object_coordinate fields to the paddded image.
  - Add "unpadded_image_size" which contains the image size of the
    augmented but unpadded image. This can be used to scale the bbox for
    visualization on the processed image.

  Padding gets added on bottom and right.

  Required fields in config:
    inputs: names of applicable fields in the example.
    target_size: (height, width) tuple.
    background_val: optional list of background pixel value. Default to 0.3.
    object_coordinate_keys: optional list of strings. Keys for coordinate
      features that describe objects, e.g. 'bbox' for object detection.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    num_inputs = len(self.config.inputs)
    backgrnd_val = self.config.get('background_val', [0.3] * num_inputs)
    target_size = self.config.target_size
    for k, backgrnd_val_ in zip(self.config.inputs, backgrnd_val):
      unpadded_image = example[k]
      if k == 'image':
        example['unpadded_image_size'] = tf.shape(unpadded_image)[-3:-1]
        height = tf.shape(unpadded_image)[0]
        width = tf.shape(unpadded_image)[1]
      example[k] = backgrnd_val_ + tf.image.pad_to_bounding_box(
          unpadded_image - backgrnd_val_, 0, 0, target_size[0], target_size[1])

    # Adjust the coordinate fields.
    object_coordinate_keys = self.config.get('object_coordinate_keys', [])
    if object_coordinate_keys:
      assert 'image' in self.config.inputs
      hratio = tf.cast(height, tf.float32) / tf.cast(target_size[0], tf.float32)
      wratio = tf.cast(width, tf.float32) / tf.cast(target_size[1], tf.float32)
      scale = tf.stack([hratio, wratio])
      for key in object_coordinate_keys:
        example[key] = data_utils.flatten_points(
            data_utils.unflatten_points(example[key]) * scale)
    return example


@TransformRegistry.register('truncate_or_pad_to_max_instances')
class TruncateOrPadToMaxInstances(Transform):
  """Truncate or pad data to a fixed length at the first (object) dimension.

  Be careful if evaluation depends on labels, truncation of gt labels may
  affect the results. If evaluation only depends on external (e.g. COCO)
  annotation, this should be fine.

  Required fields in config:
    inputs: names of applicable fields in the example.
    max_instances: positive `int` number for truncation or padding.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    max_instances = self.config.max_instances
    for k in self.config.inputs:
      truncated = example[k][:max_instances]
      example[k] = utils.pad_to_max_len(truncated, max_instances, 0)
    return example


@TransformRegistry.register('preserve_reserved_tokens')
class PreserveReservedTokens(Transform):
  """Preserve reserved tokens in points according to points_orig.

  Required fields in config:
    points_keys: names of points features.
    points_orig_keys: names of original points features.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    for points, points_orig in zip(self.config.points_keys,
                                   self.config.points_orig_keys):
      example[points] = utils.preserve_reserved_tokens(
          example[points], example[points_orig])
    return example


@TransformRegistry.register('truncate_or_pad_to_max_frames')
class TruncateOrPadToMaxFrames(Transform):
  """Truncate or pad data to a fixed number of frames.

  Required fields in config:
    inputs: names of applicable fields in the example.
    max_num_frames: positive `int` number for truncation or padding.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    max_num_frames = self.config.max_num_frames
    pad_frames = tf.maximum(0, max_num_frames - example['num_frames'])
    paddings = [[0, pad_frames], [0, 0], [0, 0], [0, 0]]

    for k in self.config.inputs:
      truncated = example[k][:max_num_frames]
      example[k] = tf.pad(truncated, paddings)
    example['num_frames'] = tf.minimum(example['num_frames'], max_num_frames)
    return example


@TransformRegistry.register('byte_encode_and_pad')
class ByteEncodeAndPad(Transform):
  """UTF-8 encoding, and truncate or pad data to a fixed number of tokens.

  Required fields in config:
    inputs: names of applicable fields in the example.
    max_len: positive `int` number for truncation or padding.
  """

  def process_example(self, example: dict[str, tf.Tensor]):
    example = copy.copy(example)
    max_len = self.config.max_len
    def to_bytes(x):
      return np.frombuffer(x, dtype=np.uint8)
      # x = tf.compat.as_bytes(x, encoding='UTF-8')
      # return tf.io.decode_raw(x, tf.uint8)
    for k in self.config.inputs:
      # Wrap inside tf.numpy_function as as_bytes doesn't accept tf.String.
      text_utf8 = tf.numpy_function(to_bytes, (example[k],), tf.uint8)
      text_utf8 = tf.cast(text_utf8, tf.int32)
      padding = tf.zeros([max_len], dtype=tf.int32)
      text_utf8 = tf.concat([text_utf8, padding], 0)[:max_len]
      text_utf8 = tf.reshape(text_utf8, [max_len])
      example[k] = text_utf8
    return example
