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
"""Pre-processing and other data utils."""

import copy
import functools
import utils
import vocab
import tensorflow as tf


def to_grayscale(image, keep_channels=True):
  image = tf.image.rgb_to_grayscale(image)
  if keep_channels:
    image = tf.tile(image, [1, 1, 3])
  return image


def random_brightness(image, max_delta, impl='simclrv2'):
  """A multiplicative vs additive change of brightness."""
  if impl == 'simclrv2':
    factor = tf.random.uniform([], tf.maximum(1.0 - max_delta, 0),
                               1.0 + max_delta)
    image = image * factor
  elif impl == 'simclrv1':
    image = tf.image.random_brightness(image, max_delta=max_delta)
  else:
    raise ValueError('Unknown impl {} for random brightness.'.format(impl))
  return image


def color_jitter_rand(image,
                      brightness=0,
                      contrast=0,
                      saturation=0,
                      hue=0,
                      impl='simclrv2'):
  """Distorts the color of the image (jittering order is random).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
    impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.

  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x):
      """Apply the i-th transformation."""
      def brightness_foo():
        if brightness == 0:
          return x
        else:
          return random_brightness(x, max_delta=brightness, impl=impl)

      def contrast_foo():
        if contrast == 0:
          return x
        else:
          return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
      def saturation_foo():
        if saturation == 0:
          return x
        else:
          return tf.image.random_saturation(
              x, lower=1-saturation, upper=1+saturation)
      def hue_foo():
        if hue == 0:
          return x
        else:
          return tf.image.random_hue(x, max_delta=hue)
      x = tf.cond(tf.less(i, 2),
                  lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                  lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
      return x

    perm = tf.random.shuffle(tf.range(4))
    for i in range(4):
      image = apply_transform(perm[i], image)
      image = tf.clip_by_value(image, 0., 1.)
    return image


def random_apply(func, p, x):
  """Randomly apply function func to x with probability p."""
  return tf.cond(
      tf.less(
          tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
          tf.cast(p, tf.float32)), lambda: func(x), lambda: x)


# from https://github.com/google-research/simclr
def random_color_jitter(image, p=1.0, strength=1.0, impl='simclrv2'):

  def _transform(image):
    color_jitter_t = functools.partial(
        color_jitter_rand, brightness=0.8*strength, contrast=0.8*strength,
        saturation=0.8*strength, hue=0.2*strength, impl=impl)
    image = random_apply(color_jitter_t, p=0.8, x=image)
    return random_apply(to_grayscale, p=0.2, x=image)
  return random_apply(_transform, p=p, x=image)


def flatten_points(points):
  """Flattens a list of point 2-tuples to a flat list.

  Dims before the last 2 dims are considered batch dims and are preserved.

  E.g. [[y1, x1], [y2, x2]] -> [y1, x1, y2, x2]

  Args:
    points: [..., N, 2]

  Returns:
    Tensor with shape [..., N * 2]
  """
  num_points = points.shape[-2]
  assert points.shape[-1] == 2
  output_shape = tf.concat([tf.shape(points)[:-2], [num_points * 2]], axis=-1)
  return tf.reshape(points, output_shape)


def unflatten_points(points):
  """Reshapes a flat list of points to a list of 2-tuples.

  Dims before the last dim are considered batch dims and are preserved.

  E.g. [y1, x1, y2, x2] -> [[y1, x1], [y2, x2]]

  Args:
    points: [..., N * 2]

  Returns:
    Tensor with shape [..., N, 2]
  """
  num_points = points.shape[-1] // 2
  output_shape = tf.concat([tf.shape(points)[:-1], [num_points, 2]], axis=-1)
  return tf.reshape(points, output_shape)


def _flip_polygons_left_right(points):
  """Left-right flip the points in polygons.

  Args:
    points: rank 1 or 2 float32 tensor containing the flat list of normalized
    y, x coords in [0, 1].

  Returns:
    Flat list of horizontally flipped points with the same rank as `points`.
  """
  points = unflatten_points(points)
  y, x = tf.split(value=points, num_or_size_splits=2, axis=-1)
  points = tf.concat([y, 1. - x], -1)
  return flatten_points(points)


def _slice(tensor, dim, start, size=None):
  """result[dim] = start[start:start+size]."""
  tensor_shape = utils.shape_as_list(tensor)
  slice_begin = [0] * tensor.shape.rank
  slice_begin[dim] = start
  slice_size = tensor_shape
  if size is not None:
    slice_size[dim] = size
  else:
    slice_size[dim] -= start
  return tf.slice(tensor, slice_begin, slice_size)


def _reverse_every(tensor, dim, n):
  while dim < 0:
    dim += tensor.shape.rank
  tshape = utils.shape_as_list(tensor)
  new_shape = tshape[:dim] + [tshape[dim] // n, n] + tshape[dim+1:]
  tensor = tf.reshape(tensor, new_shape)
  tensor = tf.reverse(tensor, [dim + 1])
  return tf.reshape(tensor, tshape)


def _flip_keypoints_left_right(points):
  """Left-right flip the keypoints.

  Args:
    points: rank 1 or 2 float32 tensor containing the flat list of normalized
    y, x coords in [0, 1].

  Returns:
    Flat list of horizontally flipped points with the same shape as `points`.
  """
  points = unflatten_points(points)
  y, x = tf.split(value=points, num_or_size_splits=2, axis=-1)
  points = tf.concat([y, 1. - x], -1)
  # The first keypoint is the nose, rest are symmetric left-right pairs.
  nose = _slice(points, dim=-2, start=0, size=1)
  all_but_nose = _slice(points, dim=-2, start=1)
  all_but_nose = _reverse_every(all_but_nose, dim=-2, n=2)
  points = tf.concat([nose, all_but_nose], axis=-2)
  return flatten_points(points)


def _flip_boxes_left_right(boxes):
  """Left-right flip the boxes.

  Args:
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4]. Boxes
      are in normalized form meaning their coordinates vary between [0, 1]. Each
      row is in the form of [ymin, xmin, ymax, xmax].

  Returns:
    Flipped boxes.
  """
  ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=-1)
  return tf.concat([ymin, 1. - xmax, ymax, 1. - xmin], -1)


def random_horizontal_flip(features, labels):
  """Randomly flips the image and boxes horizontally with probability 0.5."""
  labels, features = copy.copy(labels), copy.copy(features)
  with tf.name_scope('RandomHorizontalFlip'):
    coin_flip = tf.random.uniform([]) > 0.5
    if coin_flip:
      features['image'] = tf.image.flip_left_right(features['image'])
    # TODO(b/188693498): The order of terms in the if condition matters.
    if 'bbox' in labels and coin_flip:
      labels['bbox'] = _flip_boxes_left_right(labels['bbox'])
    if 'keypoints' in labels and coin_flip:
      labels['keypoints'] = _flip_keypoints_left_right(labels['keypoints'])
    if 'polygon' in labels and coin_flip:
      labels['polygon'] = _flip_polygons_left_right(labels['polygon'])
  return features, labels


def gather_label_indices(labels, indices, exclude_set=()):
  keys = [k for k in labels if k not in exclude_set]
  result = {k: tf.gather(labels[k], indices) for k in keys}
  for key in exclude_set:
    if key in labels:
      result[key] = labels[key]
  return result


def truncation_bbox(bbox):
  return tf.minimum(tf.maximum(bbox, 0.), 1.)


def jitter_bbox(bbox, min_range=0., max_range=0.05, truncation=True):
  """Jitter the bbox.

  Args:
    bbox: `float` tensor of shape (n, 4), ranged betwen 0 and 1.
    min_range: min jitter range in ratio to bbox size.
    max_range: max jitter range in ratio to bbox size.
    truncation: whether to truncate resultting bbox to remain [0, 1].

  Note:
    To create noisy positives, set min_range=0, which enables truncated normal
      distribution. max_range <=0.05: noisy duplicates, <=0.02: near duplicate.
    To create negatives: set min_range >= 0.1 to avoid false negatives;
      suggested max_range <=0.4 to avoid too much randomness.


  Returns:
    jittered bbox.
  """
  n = tf.shape(bbox)[0]
  h = bbox[:, 2] - bbox[:, 0]
  w = bbox[:, 3] - bbox[:, 1]
  noise = tf.stack([h, w, h, w], -1)
  if min_range == 0:
    noise_rate = tf.random.truncated_normal(
        [n, 4], mean=0, stddev=max_range/2., dtype=bbox.dtype)
  else:
    noise_rate1 = tf.random.uniform([n, 4], min_range, max_range)
    noise_rate2 = tf.random.uniform([n, 4], -max_range, -min_range)
    selector = tf.cast(tf.random.uniform([n, 4], 0, 1) < 0.5, tf.float32)
    noise_rate = noise_rate1 * selector + noise_rate2 * (1. - selector)
  bbox = bbox + noise * noise_rate
  return truncation_bbox(bbox) if truncation else bbox


def shift_bbox(bbox, truncation=True):
  """Shifting bbox without changing the bbox height and width."""
  n = tf.shape(bbox)[0]
  # randomly sample new bbox centers.
  cy = tf.random.uniform([n, 1], 0, 1)
  cx = tf.random.uniform([n, 1], 0, 1)
  h = bbox[:, 2:3] - bbox[:, 0:1]
  w = bbox[:, 3:4] - bbox[:, 1:2]
  bbox = tf.concat([cy - tf.abs(h)/2, cx - tf.abs(w)/2,
                    cy + tf.abs(h)/2, cx + tf.abs(w)/2], -1)
  return truncation_bbox(bbox) if truncation else bbox


def random_bbox(n, max_size=1.0, truncation=True):
  """Generating random n bbox with max size specified within [0, 1]."""
  cy = tf.random.uniform([n, 1], 0, 1)
  cx = tf.random.uniform([n, 1], 0, 1)
  h = tf.random.truncated_normal([n, 1], 0, max_size/2.)
  w = tf.random.truncated_normal([n, 1], 0, max_size/2.)
  bbox = tf.concat([cy - tf.abs(h)/2, cx - tf.abs(w)/2,
                    cy + tf.abs(h)/2, cx + tf.abs(w)/2], -1)
  return truncation_bbox(bbox) if truncation else bbox


def augment_bbox(bbox, bbox_label, max_jitter, n_noise_bbox, mix_rate=0.):
  """Augment bbox.

  There are two types of noises to add:
    1. Bad bbox: jittered bbox, shifted bbox, or random bbox.
    2. Duplicated bbox.

  Args:
    bbox: `float` tensor of shape (n, 4), ranged betwen 0 and 1.
    bbox_label: `int` tensor of shape (n,).
    max_jitter: `float` scalar specifying max jitter range for positive bbox.
    n_noise_bbox: `int` scalar tensor specifying size of the extra noise to add.
    mix_rate: `float`. Probability of injecting the bad bbox in the middle of
      original bbox, followed by dup bbox at the end; otherwise simply append
      all noises at the end of original bbox.

  Returns:
    bbox_new: augmented bbox that's `n_noise_bbox` larger than original.
    label_new: new label for bbox_new.
    is_real: a `float` 0/1 indicator for whether a bbox is real.
    is_noise: a `float` 0/1 indicator for whether a bbox is extra.
  """
  n = tf.shape(bbox)[0]
  dup_bbox_size = tf.random.uniform(
      [], 0, n_noise_bbox + 1, dtype=tf.int32)
  dup_bbox_size = 0 if n == 0 else dup_bbox_size
  bad_bbox_size = n_noise_bbox - dup_bbox_size
  multiplier = 1 if n == 0 else tf.math.floordiv(n_noise_bbox, n) + 1
  bbox_tiled = tf.tile(bbox, [multiplier, 1])

  # Create bad bbox.
  bbox_tiled = tf.random.shuffle(bbox_tiled)
  bad_bbox_shift = shift_bbox(bbox_tiled[:bad_bbox_size], truncation=True)
  bad_bbox_random = random_bbox(bad_bbox_size, max_size=1.0, truncation=True)
  bad_bbox = tf.concat([bad_bbox_shift, bad_bbox_random], 0)
  bad_bbox = tf.random.shuffle(bad_bbox)[:bad_bbox_size]
  bad_bbox_label = tf.zeros([bad_bbox_size], dtype=bbox_label.dtype) + (
      vocab.FAKE_CLASS_TOKEN - vocab.BASE_VOCAB_SHIFT)

  # Create dup bbox.
  bbox_tiled = tf.random.shuffle(bbox_tiled)
  dup_bbox = jitter_bbox(
      bbox_tiled[:dup_bbox_size], min_range=0, max_range=0.1, truncation=True)
  dup_bbox_label = tf.zeros([dup_bbox_size], dtype=bbox_label.dtype) + (
      vocab.FAKE_CLASS_TOKEN - vocab.BASE_VOCAB_SHIFT)

  # Jitter positive bbox.
  if max_jitter > 0:
    bbox = jitter_bbox(bbox, min_range=0, max_range=max_jitter, truncation=True)

  if tf.random.uniform([]) < mix_rate:
    # Mix the bbox with bad bbox, appneded by dup bbox.
    bbox_new = tf.concat([bbox, bad_bbox], 0)
    bbox_new_label = tf.concat([bbox_label, bad_bbox_label], 0)
    idx = tf.random.shuffle(tf.range(tf.shape(bbox_new)[0]))
    bbox_new = tf.gather(bbox_new, idx)
    bbox_new_label = tf.gather(bbox_new_label, idx)
    bbox_new = tf.concat([bbox_new, dup_bbox], 0)
    bbox_new_label = tf.concat([bbox_new_label, dup_bbox_label], 0)
  else:
    # Merge bad bbox and dup bbox into noise bbox.
    noise_bbox = tf.concat([bad_bbox, dup_bbox], 0)
    noise_bbox_label = tf.concat([bad_bbox_label, dup_bbox_label], 0)

    if n_noise_bbox > 0:
      idx = tf.random.shuffle(tf.range(n_noise_bbox))
      noise_bbox = tf.gather(noise_bbox, idx)
      noise_bbox_label = tf.gather(noise_bbox_label, idx)

    # Append noise bbox to bbox and create mask.
    bbox_new = tf.concat([bbox, noise_bbox], 0)
    bbox_new_label = tf.concat([bbox_label, noise_bbox_label], 0)

  return bbox_new, bbox_new_label


def inject_noise_bbox(labels, max_instances_per_image):
  labels = copy.copy(labels)
  num_instances = tf.shape(labels['bbox'])[0]
  if num_instances < max_instances_per_image:
    n_noise_bbox = max_instances_per_image - num_instances
    labels['bbox'], labels['label'] = augment_bbox(
        labels['bbox'], labels['label'], 0., n_noise_bbox)
  return labels


def reorder_object_instances(features, labels, order):
  """Must be called _before_ padding to max instances."""
  if order == 'none':
    return features, labels

  bbox = labels['bbox']
  assert bbox.shape.rank == 2, 'Must be unbatched'
  bbox = tf.reshape(bbox, [-1, 2, 2])

  if order == 'random':
    idx = tf.random.shuffle(tf.range(tf.shape(bbox)[0]))
  elif order == 'area':
    areas = tf.cast(tf.reduce_prod(bbox[:, 1, :] - bbox[:, 0, :], axis=1),
                    tf.int64)  # approximated size.
    idx = tf.argsort(areas, direction='DESCENDING')
  elif order == 'dist2ori':
    y, x = bbox[:, 0], bbox[:, 1]  # using top-left corner.
    dist2ori = tf.square(y) + tf.square(x)
    idx = tf.argsort(dist2ori, direction='ASCENDING')
  else:
    raise ValueError('Unknown order {}'.format(order))

  labels = gather_label_indices(labels, idx)

  return features, labels


def random_resize(image, min_image_size, max_image_size, max_out_prob=0.,
                  resize_method=tf.image.ResizeMethod.BILINEAR):
  """Randomly resize the image while preserving aspect ratio.

  Args:
    image: 3D image tensor.
    min_image_size: minimum size for the longer side of the image.
    max_image_size: maximum size for the longer side of the image.
    max_out_prob: probability of being max image size.
    resize_method: one of tf.image.ResizeMethod.

  Returns:
    resized image.
  """
  if tf.random.uniform([]) < max_out_prob:
    size_longer_side = max_image_size
  else:
    size_longer_side = tf.random.uniform(
        [], min_image_size, max_image_size+1, dtype=tf.int32)

  h, w = tf.unstack(tf.shape(image)[:2])
  if w > h:
    ow = size_longer_side
    oh = tf.cast(size_longer_side * h / w, tf.int32)
  else:
    oh = size_longer_side
    ow = tf.cast(size_longer_side * w / h, tf.int32)
  return tf.image.resize(image, (oh, ow), method=resize_method)


def filter_invalid_objects(labels, filter_crowd=False):
  """Filtering out objects that are invalid/undesirable."""
  # TODO(iamtingchen): filtering other invalid objects (e.g. no keypoint left).
  # Filtering out invalid bbox.
  bbox = labels['bbox']
  box_valid = tf.logical_and(bbox[:, 2] > bbox[:, 0], bbox[:, 3] > bbox[:, 1])
  # Filtering out crowded objects.
  if filter_crowd:
    not_crowd = tf.logical_not(labels['is_crowd'])
    valid_indices = tf.where(tf.logical_and(box_valid, not_crowd))[:, 0]
  else:
    valid_indices = tf.where(box_valid)[:, 0]
  return gather_label_indices(labels, valid_indices)


def scale_jitter(image, min_scale, max_scale, target_height, target_width,
                 interp=tf.image.ResizeMethod.BILINEAR):
  """Scale the image to a random scale of target size (keeping aspect ratio)."""
  input_size = tf.cast(tf.shape(image)[:2], tf.float32)
  output_size = tf.constant([target_height, target_width], tf.float32)
  random_scale = tf.random.uniform([], min_scale, max_scale)
  random_scale_size = tf.multiply(output_size, random_scale)
  scale = tf.minimum(
      random_scale_size[0] / input_size[0], random_scale_size[1] / input_size[1]
  )
  scaled_size = tf.cast(tf.multiply(input_size, scale), tf.int32)
  scaled_image = tf.image.resize(
      image, tf.cast(scaled_size, tf.int32), method=interp)
  return scaled_image


def fixed_size_crop(features, labels, target_height, target_width,
                    object_coordinate_keys):
  """Crops a random region of target size from the image."""
  input_size = tf.shape(features['image'])[:2]
  output_size = tf.stack([target_height, target_width])

  max_offset = tf.subtract(input_size, output_size)
  max_offset = tf.cast(tf.maximum(max_offset, 0), tf.float32)
  offset = tf.multiply(max_offset, tf.random.uniform([], 0.0, 1.0))
  offset = tf.cast(offset, tf.int32)

  region = (offset[0], offset[1],
            tf.minimum(output_size[0], input_size[0] - offset[0]),
            tf.minimum(output_size[1], input_size[1] - offset[1]))

  return crop(features, labels, region, object_coordinate_keys)


def random_crop(features, labels, scale, ratio, object_coordinate_keys):
  """Crops image to random aspect ratio.

  Args:
    features: `dict` containing image tensor.
    labels: `dict` containing label tensors such as `bbox`.
    scale: tuple of (min, max) range of scaling factor of the crop (<= 1.0).
    ratio: tuple of (min, max) range of aspect ratio of the crop.
    object_coordinate_keys: a tuple of name keys for coordinate labels.

  Returns:
    cropped image, cropped bbox
  """
  image = features['image']
  assert scale[0] <= 1.0, scale[1] <= 1.0
  img_size = tf.unstack(tf.shape(image))
  area = tf.cast(img_size[1] * img_size[0], tf.float32)

  target_area = tf.random.uniform([], *scale, dtype=tf.float32) * area

  log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))
  aspect_ratio = tf.math.exp(
      tf.random.uniform([], *log_ratio, dtype=tf.float32))

  h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)
  w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
  h = tf.minimum(h, img_size[0])
  w = tf.minimum(w, img_size[1])

  h_offset = tf.random.uniform([], 0, img_size[0] - h + 1, dtype=tf.int32)
  w_offset = tf.random.uniform([], 0, img_size[1] - w + 1, dtype=tf.int32)

  region = (h_offset, w_offset, h, w)

  return crop(features, labels, region, object_coordinate_keys)


def crop(features, labels, region, object_coordinate_keys):
  """Crop image to region and adjust (normalized) bbox."""
  image = features['image']
  h_offset, w_offset, h, w = region
  h_ori, w_ori, _ = tf.unstack(tf.shape(image))

  features['image'] = image[h_offset:h_offset + h, w_offset:w_offset + w, :]

  h_offset = tf.cast(h_offset, tf.float32)
  w_offset = tf.cast(w_offset, tf.float32)
  h, w = tf.cast(h, tf.float32), tf.cast(w, tf.float32)
  h_ori, w_ori = tf.cast(h_ori, tf.float32), tf.cast(w_ori, tf.float32)

  scale = tf.stack([h_ori / h, w_ori / w])
  offset = tf.stack([h_offset / h_ori, w_offset / w_ori])
  for key in object_coordinate_keys:
    if key in labels:
      points = labels[key]
      points = unflatten_points(points)
      points = (points - offset) * scale
      points = tf.math.minimum(tf.math.maximum(points, 0), 1)
      labels[key] = flatten_points(points)

  return features, labels


def pad_image_to_max_size(features,
                          labels,
                          max_image_size,
                          object_coordinate_keys,
                          backgrnd_val=0.3,):  # pylint: disable=g-doc-args
  """Pad image to the same size (max_image_size, max_image_size, 3).

  - Replace labels["image"] with padded image.
  - Re-scale labels to the paddded image.
  - Add "unpadded_image_size" to labels which contains the image size of the
    augmented but unpadded image. This can be used to scale the bbox for
    visualization on the processed image.

  Padding gets added on bottom and right.

  Returns:
    features, labels
  """
  unpadded_image = features['image']
  features['unpadded_image_size'] = tf.shape(unpadded_image)[:2]

  features['image'] = backgrnd_val + tf.image.pad_to_bounding_box(
      unpadded_image - backgrnd_val, 0, 0, max_image_size, max_image_size)

  height = tf.shape(unpadded_image)[0]
  width = tf.shape(unpadded_image)[1]
  hratio = tf.cast(height, tf.float32) / tf.cast(max_image_size, tf.float32)
  wratio = tf.cast(width, tf.float32) / tf.cast(max_image_size, tf.float32)
  scale = tf.stack([hratio, wratio])
  for key in object_coordinate_keys:
    if key in labels:
      labels[key] = flatten_points(
          unflatten_points(labels[key]) * scale)

  return features, labels


def truncate_or_pad_to_max_instances(labels, max_instances):
  """Truncate or pad data to a fixed length at the first (object) dimension.

  Be careful if evaluation depends on labels, truncation of gt labels may
  affect the results. If evaluation only depends on external (COCO) annotation,
  this should be fine.

  Args:
    labels: `dict` of label tensors where the first dimension of each tensor is
      associated with object instances. For example, bbox is of shape
      (num_instances, 4).
    max_instances: positive `int` number for truncation or padding.

  Returns:
    labels with the same first dimension size.
  """
  for k in labels:
    truncated = labels[k][:max_instances]
    labels[k] = utils.pad_to_max_len(truncated, max_instances, 0)
  return labels


def preprocess_train(features,
                     labels,
                     max_image_size,
                     max_instances_per_image,
                     object_order=None,
                     inject_noise_instances=False,
                     jitter_scale=(0.5, 1.6),
                     random_flip=True,
                     color_jitter_strength=0.,
                     filter_invalid_labels=True,
                     object_coordinate_keys=('bbox', 'polygon', 'keypoints')):
  """Preprocessing for training input pipeline (scale jittering-based)."""
  if features['image'].dtype != tf.float32:
    features['image'] = tf.image.convert_image_dtype(
        features['image'], tf.float32)
  features['image'] = scale_jitter(
      features['image'], jitter_scale[0], jitter_scale[1],
      max_image_size, max_image_size)
  features, labels = fixed_size_crop(
      features, labels, max_image_size, max_image_size, object_coordinate_keys)
  if random_flip:
    features, labels = random_horizontal_flip(features, labels)
  if color_jitter_strength > 0:
    features['image'] = random_color_jitter(
        features['image'], strength=color_jitter_strength, impl='simclrv2')
    features['image'] = tf.clip_by_value(features['image'], 0., 1.)
  if filter_invalid_labels:
    labels = filter_invalid_objects(labels, filter_crowd=True)
  if object_order is not None:  # for detection.
    features, labels = reorder_object_instances(features, labels, object_order)
  if inject_noise_instances:  # for detection.
    labels = inject_noise_bbox(labels, max_instances_per_image)
  features, labels = pad_image_to_max_size(
      features, labels, max_image_size, object_coordinate_keys)
  if max_instances_per_image > 0:
    labels = truncate_or_pad_to_max_instances(labels, max_instances_per_image)
  return features, labels


def preprocess_eval(features,
                    labels,
                    max_image_size,
                    max_instances_per_image,
                    object_coordinate_keys=('bbox', 'polygon', 'keypoints')):
  """Preprocessing for eval input pipeline."""
  features['image'] = random_resize(
      features['image'], max_image_size, max_image_size, max_out_prob=1.0)
  features, labels = pad_image_to_max_size(
      features, labels, max_image_size, object_coordinate_keys)
  if max_instances_per_image > 0:
    labels = truncate_or_pad_to_max_instances(labels, max_instances_per_image)
  return features, labels

