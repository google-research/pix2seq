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
"""ConvNet blocks."""

import math
import numpy as np
import tensorflow as tf
import tensorflow_addons.layers as tfa_layers


def get_variable_initializer(scale=1e-10):
  return tf.keras.initializers.VarianceScaling(
      scale=scale, mode='fan_avg', distribution='uniform')


def get_norm(norm_type, **kwargs):
  """Normalization Layer."""
  if norm_type == 'group_norm':
    return tfa_layers.GroupNormalization(
        groups=kwargs.get('num_groups', 32),
        axis=-1,
        epsilon=1e-5,
        name=kwargs.get('name', 'group_nrom'))
  elif norm_type == 'layer_norm':
    return tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
        name=kwargs.get('name', 'layer_norm'))
  elif norm_type == 'none':
    return tf.identity
  else:
    raise ValueError(f'Unknown normalization {norm_type}')


class UpsampleBlock(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               dim,
               upsample='none',
               upsample_factor=2,
               **kwargs):
    super().__init__(**kwargs)
    self.dim = dim
    self.upsample = upsample
    self.upsample_factor = upsample_factor

  def build(self, input_shapes):
    dim_hr = self.dim if input_shapes[0] is None else input_shapes[0][-1]
    if self.upsample == 'none':
      self.upsample = lambda x: x
    else:
      self.upsample = tf.keras.layers.UpSampling2D(
          size=(self.upsample_factor, self.upsample_factor),
          interpolation=self.upsample)

    self.conv1 = tf.keras.layers.Conv2D(
        filters=dim_hr,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding='SAME',
        use_bias=True,
        kernel_initializer=get_variable_initializer(1.0),
        name='conv1')

    self.gn1 = get_norm(
        'group_norm',
        num_groups=min(dim_hr // 4, 32),
        name='group_norm1')

    self.conv2 = tf.keras.layers.Conv2D(
        filters=self.dim,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding='SAME',
        use_bias=True,
        kernel_initializer=get_variable_initializer(1.0),
        name='conv2')

    self.gn2 = get_norm(
        'group_norm',
        num_groups=min(self.dim // 4, 32),
        name='group_norm2')

  def call(self, x_hr_n_lr, training):
    x_hr = 0. if x_hr_n_lr[0] is None else x_hr_n_lr[0]
    x_lr = tf.nn.silu(self.gn1(self.conv1(x_hr_n_lr[1])))
    return tf.nn.silu(self.gn2(self.conv2(x_hr + self.upsample(x_lr))))
    # return x_hr + self.upsample(x_lr)


class FeaturePyramidMerge(tf.keras.layers.Layer):  # pylint: disable=missing-docstring
  """Upsample each lr to hr, and finally upsample to the out_size.

  Each upsample factor is determined by target_size // source_size. For the
  final upsample to out_size, if `final_gradual_upsample=True`, it will do so
  with a series of factor 2 upsamplings, otherwise it will be one shot upsample.
  """

  def __init__(self,
               out_dim,
               out_size,
               upsample='nearest',
               final_gradual_upsample=False,
               **kwargs):
    super().__init__(**kwargs)
    self.out_dim = out_dim
    self.out_size = out_size
    self.upsample = upsample
    self.final_gradual_upsample = final_gradual_upsample

  def build(self, input_shapes):
    self.n_feature_layers = len(input_shapes)
    self.upsample_blocks = {}
    for i in range(self.n_feature_layers)[::-1]:
      if i != 0:
        upsample = self.upsample
        if input_shapes[i][1:3] == input_shapes[i-1][1:3]:
          upsample = 'none'
        if (input_shapes[i-1][1] > self.out_size[0]) or (
            input_shapes[i-1][2] > self.out_size[1]):
          # If high-res map is larger than out_size, just use the low-res one.
          self.upsample_blocks[str(i)] = lambda x, _: x[1]
        else:
          self.upsample_blocks[str(i)] = UpsampleBlock(
              dim=input_shapes[i-1][-1],
              upsample=upsample,
              upsample_factor=input_shapes[i-1][1] // input_shapes[i][1],
              name=f'upsample_block_{i}')
    if self.final_gradual_upsample:
      self.n_solo_up_layers = int(np.log2(self.out_size // input_shapes[0][1]))
      for i in range(self.n_solo_up_layers):
        self.upsample_blocks[f'solo{i}'] = UpsampleBlock(
            dim=self.out_dim,
            upsample=self.upsample,
            upsample_factor=2,
            name=f'upsample_block_solo_{i}')
    else:
      factor_h = self.out_size[0] // input_shapes[0][1]
      factor_w = self.out_size[1] // input_shapes[0][2]
      assert factor_h == factor_w, (self.out_size, input_shapes)
      if factor_h > 1 or factor_w > 1:
        # Skip final upsampling if the largest feature map has the same size
        # as out_size.
        self.upsample_blocks['0'] = UpsampleBlock(
            dim=self.out_dim,
            upsample=self.upsample,
            upsample_factor=factor_h,
            name='upsample_block_0')
      else:
        self.upsample_blocks['0'] = lambda x, _: x[1]

  def call(self, h_stack, training):
    """call function.

    Args:
      h_stack: a list of 4d feature maps of (bsz, h, w, c) from high res to low
        res.
      training: if it is in training mode.

    Returns:
      a single feature map of (bsz, out_size, out_size, out_dim)
    """
    x = h_stack[-1]
    for i in range(self.n_feature_layers)[::-1]:
      if i != 0:
        x = self.upsample_blocks[str(i)]([h_stack[i-1], x], training)
    if self.final_gradual_upsample:
      for i in range(self.n_solo_up_layers):
        x = self.upsample_blocks[f'solo{i}']([None, x], training)
    else:
      x = self.upsample_blocks['0']([None, x], training)
    return x


class FeaturePyramidMergeNaive(tf.keras.layers.Layer):  # pylint: disable=missing-docstring
  """Upsample every feature maps to out_size, followed by a single conv."""

  def __init__(self,
               out_dim,
               out_size,
               upsample='nearest',
               **kwargs):
    super().__init__(**kwargs)
    self.out_dim = out_dim
    self.out_size = out_size
    self.upsample = upsample

  def build(self, input_shapes):
    self.n_feature_layers = len(input_shapes)
    self.upsamples = {}
    for i in range(self.n_feature_layers):
      up_factor = self.out_size // input_shapes[i][1]
      self.upsamples[str(i)] = tf.keras.layers.UpSampling2D(
          size=(up_factor, up_factor), interpolation=self.upsample)

    self.conv = tf.keras.layers.Conv2D(
        filters=self.out_dim,
        kernel_size=[1, 1],  # could be 3x3
        strides=[1, 1],
        padding='SAME',
        use_bias=True,
        kernel_initializer=get_variable_initializer(1.0),
        name='conv')

    self.gn = get_norm(
        'group_norm',
        num_groups=min(self.out_dim // 4, 32),
        name='group_norm')

  def call(self, h_stack, training):
    h_stack_new = []
    for i in range(len(h_stack)):
      h_stack_new.append(self.upsamples[str(i)](h_stack[i]))
    return tf.nn.silu(self.gn(self.conv(tf.concat(h_stack_new, -1))))


class DepthwiseConvBlock(tf.keras.layers.Layer):  # pylint: disable=missing-docstring
  """Depthwise conv followed by pointwise/1x1 conv."""

  def __init__(self,
               out_dim,
               kernel_size,
               dropout_rate=0.,
               **kwargs):
    super().__init__(**kwargs)
    self._out_dim = out_dim
    self._kernel_size = kernel_size
    self._dropout_rate = dropout_rate

  def build(self, input_shapes):
    input_dim = self._out_dim if input_shapes is None else input_shapes[-1]
    self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=self._kernel_size,
        padding='SAME',
        use_bias=True,
        kernel_initializer=get_variable_initializer(1.0),
        name='depthwise_conv')
    self.gn = get_norm(
        'group_norm',
        num_groups=min(input_dim // 4, 32),
        name='gn')
    self.dropout = tf.keras.layers.Dropout(self._dropout_rate)
    self.pointwise_conv = tf.keras.layers.Conv2D(
        filters=self._out_dim,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding='SAME',
        use_bias=True,
        kernel_initializer=get_variable_initializer(1e-10),
        name='pointwise_conv')

  def call(self, x, training, size=None):
    """call function.

    Args:
      x: `Tensor` of (bsz, h, w, c) or (bsz, seqlen, c).
      training: `Boolean` indicator.
      size: set to None if x is an 4d image tensor, otherwise set size=h*w,
        where seqlen=size+extra, and conv is performed only on the first part.

    Returns:
      `Tensor` of the same shape as input x.
    """
    x_skip = x
    if size is not None:  # Resize sequence into an image for 2d conv.
      x_skip = x[:, :size]
      x_remain = x[:, size:]
      height = width = int(math.sqrt(size))
      x = tf.reshape(x_skip, [tf.shape(x)[0], height, width, tf.shape(x)[-1]])
    x = tf.nn.silu(self.gn(self.depthwise_conv(x)))
    x = self.dropout(x, training=training)
    x = self.pointwise_conv(x)
    # TODO(iamtingchen): consider unet style ordering of gn&conv.
    # x = tf.nn.silu(self.gn1(x))
    # x = self.depthwise_conv(x)
    # x = tf.nn.silu(self.gn2(x))
    # x = self.dropout(x, training=training)
    # x = self.pointwise_conv(x)
    # TODO(iamtingchen): consider transformer/normer style ordering of gn&conv.
    # x = tf.nn.silu(self.depthwise_conv(self.gn1(x)))
    # x = self.dropout(x, training=training)
    # x = self.pointwise_conv(self.gn2(x))
    if size is not None:
      x = x_skip + tf.reshape(x, [tf.shape(x)[0], size, tf.shape(x)[-1]])
      x = tf.concat([x, x_remain], 1)
    else:
      x = x_skip + x
    return x
