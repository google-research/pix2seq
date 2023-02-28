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
"""TransUNet architectures."""

import ml_collections

from architectures.transformers import add_vis_pos_emb
from architectures.transformers import get_shape
from architectures.transformers import positional_encoding
from architectures.transformers import TransformerDecoder
from utils import int2bits

import tensorflow as tf
import tensorflow_addons.layers as tfa_layers


def scaled_sum(inputs):
  return sum(inputs) / tf.math.sqrt(float(len(inputs)))


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


class ScalarEmbedding(tf.keras.layers.Layer):
  """Scalar embedding layers.

  Assume the first input dim to be time, and rest are optional features.
  """

  def __init__(self, dim, scaling, expansion=4, **kwargs):
    super().__init__(**kwargs)
    self.scalar_encoding = lambda x: positional_encoding(x*scaling, dim)
    self.dense_0 = tf.keras.layers.Dense(
        dim * expansion,
        kernel_initializer=get_variable_initializer(1.),
        name='dense0')
    self.dense_1 = tf.keras.layers.Dense(
        dim * expansion,
        kernel_initializer=get_variable_initializer(1.),
        name='dense1')

  def call(self, x, last_swish=True, normalize=False):
    y = None
    if x.shape.rank > 1:
      assert x.shape.rank == 2
      x, y = x[..., 0], x[..., 1:]
    x = self.scalar_encoding(x)[0]
    if normalize:
      x_mean = tf.reduce_mean(x, -1, keepdims=True)
      x_std = tf.math.reduce_std(x, -1, keepdims=True)
      x = (x - x_mean) / x_std
    x = tf.nn.silu(self.dense_0(x))
    x = x if y is None else tf.concat([x, y], -1)
    x = self.dense_1(x)
    return tf.nn.silu(x) if last_swish else x


class ResnetBasicBlock(tf.keras.layers.Layer):
  """Resnet Block."""

  def __init__(self,
               out_dim,
               kernel_size,
               dropout,
               **kwargs):
    super().__init__(**kwargs)
    self.out_dim = out_dim
    self.kernel_size = kernel_size
    self.dropout_rate = dropout

  def build(self, input_shape):
    if self.out_dim == -1:
      self.out_dim = input_shape[-1]
    if self.out_dim != input_shape[-1]:
      self.shortcut_layer = tf.keras.layers.Conv2D(
          filters=self.out_dim,
          kernel_size=[1, 1],
          strides=[1, 1],
          padding='SAME',
          use_bias=True,
          kernel_initializer=get_variable_initializer(1.0),
          name='shortcut_layer')
    else:
      self.shortcut_layer = tf.identity
    self.emb_proj = tf.keras.layers.Dense(
        units=self.out_dim * 2,
        kernel_initializer=get_variable_initializer(1.0),
        name='emb_proj')
    self.conv_0 = tf.keras.layers.Conv2D(
        filters=self.out_dim,
        kernel_size=[self.kernel_size, self.kernel_size],
        strides=[1, 1],
        padding='SAME',
        use_bias=True,
        kernel_initializer=get_variable_initializer(1.0),
        name='conv_0')
    self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
    self.conv_1 = tf.keras.layers.Conv2D(
        filters=self.out_dim,
        kernel_size=[self.kernel_size, self.kernel_size],
        strides=[1, 1],
        padding='SAME',
        use_bias=True,
        kernel_initializer=get_variable_initializer(),
        name='conv_1')
    self.group_norm_0 = get_norm(
        'group_norm',
        num_groups=min(input_shape[-1] // 4, 32),
        name='group_norm_0')
    self.group_norm_1 = get_norm(
        'group_norm',
        num_groups=min(self.out_dim // 4, 32),
        name='group_norm_1')

  def call(self, x, emb, training):
    """x in [bsz, h, w, c], emb in [bsz, c']."""
    skip = self.shortcut_layer(x)

    if emb is None:
      emb_scale, emb_shift = 1.0, 0.
    else:
      emb = self.emb_proj(emb)[:, None, None, :]
      emb_scale, emb_shift = tf.split(emb, 2, axis=-1)
      emb_scale += 1.0

    x = tf.nn.silu(self.group_norm_0(x))
    x = self.conv_0(x)
    x = tf.nn.silu(self.group_norm_1(x) * emb_scale + emb_shift)
    x = self.dropout(x, training=training)
    x = self.conv_1(x)
    return scaled_sum([x, skip])


class ConvUnit(tf.keras.layers.Layer):
  """Conv computation unit.

  For bottom-up (downsampling) blocks:
    x n
      ResNet blocks
    (optional) multi-head self-attention
    (optional) Downsampling conv

  For top-down (upsampling) blocks:
    x n
      Merging lateral connection
      ResNet blocks
    (optional) multi-head self-attention
    (optional) Upsampling conv
  """

  def __init__(self,
               config,
               n_res_blocks=3,
               kernel_size=3,
               ch_multiplier=1,
               expect_lateral=False,
               up_dn_sample='none',
               **kwargs):
    super().__init__(**kwargs)
    self.config = config
    self.n_res_blocks = n_res_blocks
    self.kernel_size = kernel_size
    self.ch_multiplier = ch_multiplier
    self.up_dn_sample = up_dn_sample
    self.expect_lateral = expect_lateral

  def build(self, input_shape):
    config = self.config
    out_dim = self.ch_multiplier * config.dim

    if self.up_dn_sample == 'dn':
      self.conv_dn = tf.keras.layers.Conv2D(
          filters=out_dim,
          kernel_size=[3, 3],
          strides=[2, 2],
          padding='SAME',
          use_bias=True,
          kernel_initializer=get_variable_initializer(1.0),
          name='conv_dn')
    elif self.up_dn_sample == 'up':
      self.upsample = tf.keras.layers.UpSampling2D(
          size=(2, 2), interpolation='nearest')
      self.conv_up = tf.keras.layers.Conv2D(
          filters=out_dim,
          kernel_size=[3, 3],
          strides=[1, 1],
          padding='SAME',
          use_bias=True,
          kernel_initializer=get_variable_initializer(1.0),
          name='conv_up')
    else:
      assert self.up_dn_sample == 'none'

    self.resnet_blocks = {}
    self.mhca_blocks = {}
    self.mhca_norms = {}
    self.mhsa_blocks = {}
    self.mhsa_norms = {}
    for i in range(self.n_res_blocks):
      self.resnet_blocks[str(i)] = ResnetBasicBlock(
          out_dim=out_dim,
          kernel_size=self.kernel_size,
          dropout=config.dropout,
          name=f'res_block_{i}')

      self.do_attn = input_shape[1] in config.mhsa_resolutions
      if self.do_attn:
        self.mhsa_norms[str(i)] = get_norm(
            config.norm_type,
            name=f'mhsa_norms_{i}')
        self.mhsa_blocks[str(i)] = tf.keras.layers.MultiHeadAttention(
            num_heads=out_dim // config.per_head_dim,
            key_dim=config.per_head_dim,
            value_dim=None,
            dropout=0.,
            name=f'mhsa_block_{i}')

        if config.conditioning:
          self.mhca_norms[str(i)] = get_norm(
              config.norm_type,
              name=f'mhca_norms_{i}')
          self.mhca_blocks[str(i)] = tf.keras.layers.MultiHeadAttention(
              num_heads=out_dim // config.per_head_dim,
              key_dim=config.per_head_dim,
              value_dim=None,
              dropout=0.,
              name=f'mhca_block_{i}')

  def call(self, x, emb, cembs, x_list, training):
    for i in range(self.n_res_blocks):
      # merge lateral connection.
      if self.expect_lateral:
        x = tf.concat([x, x_list.pop()], -1)  # alternative, but more params.
      # resnet blocks.
      x = self.resnet_blocks[str(i)](x, emb, training)
      # save for lateral connection
      if not self.expect_lateral:
        x_list.append(x)

      if self.do_attn:
        bsz, h, w, c = get_shape(x)
        x = tf.reshape(x, [bsz, h * w, c])
        if self.mhca_blocks:
          x_p = self.mhca_norms[str(i)](x)
          x_p = self.mhca_blocks[str(i)](x_p, cembs, cembs, training=training)
          x = scaled_sum([x, x_p])
        if self.mhsa_blocks:
          x_p = self.mhsa_norms[str(i)](x)
          x_p = self.mhsa_blocks[str(i)](x_p, x_p, x_p, training=training)
          x = scaled_sum([x, x_p])
        x = tf.reshape(x, [bsz, h, w, c])

    if self.up_dn_sample == 'dn':
      x = self.conv_dn(x)
      x_list.append(x)
    elif self.up_dn_sample == 'up':
      x = self.conv_up(self.upsample(x))
    return x


class Transformer(tf.keras.layers.Layer):
  """Transformer."""

  def __init__(self,
               dim=256,
               strides=2,
               num_layers=2,
               num_heads=4,
               mlp_ratio=4,
               drop_units=0.1,
               drop_att=0.,
               drop_path=0.,
               pos_encoding='sin_cos',
               conditioning=True,
               **kwargs):
    super().__init__(**kwargs)
    self.dim = dim
    self.strides = strides
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.mlp_ratio = mlp_ratio
    self.drop_units = drop_units
    self.drop_att = drop_att
    self.drop_path = drop_path
    self.pos_encoding = pos_encoding
    self.conditioning = conditioning

  def build(self, input_shape):
    strides = self.strides
    _, height, width, _ = input_shape

    self.trans_conv_in = tf.keras.layers.Conv2D(
        filters=self.dim,
        kernel_size=strides,
        strides=strides,
        padding='VALID',
        use_bias=True,
        kernel_initializer=get_variable_initializer(1.0),
        name='trans_conv_in')
    self.trans_conv_in_norm = get_norm(
        'group_norm',
        num_groups=min(self.dim // 4, 32),
        name='trans_conv_in_norm')

    n_rows, n_cols = height // strides, width // strides
    if self.pos_encoding:
      add_vis_pos_emb(self, self.pos_encoding, n_rows, n_cols, self.dim)
    self.emb_proj = tf.keras.layers.Dense(
        units=self.dim,
        kernel_initializer=get_variable_initializer(1.0),
        name='emb_proj')

    self.transformer_decoder = TransformerDecoder(
        num_layers=self.num_layers,
        dim=self.dim,
        mlp_ratio=self.mlp_ratio,
        num_heads=self.num_heads,
        drop_path=self.drop_path,
        drop_units=self.drop_units,
        drop_att=self.drop_att,
        self_attention=True,
        cross_attention=self.conditioning,
        name='transformer_decoder')

    self.trans_conv_out_norm = get_norm(
        'group_norm',
        num_groups=min(self.dim // 4, 32),
        name='trans_conv_out_norm')
    self.trans_conv_out = tf.keras.layers.Conv2D(
        filters=self.dim,
        kernel_size=1,
        strides=1,
        padding='VALID',
        use_bias=True,
        kernel_initializer=get_variable_initializer(1.0),
        name='trans_conv_out')

  def call(self, x, t, c, training):
    """Input x of (bsz, h, w, c), t in (bsz, d), optional c in (bsz, seq, d)."""
    x = self.trans_conv_in(self.trans_conv_in_norm(x))
    bsz, h, w, dim = get_shape(x)
    x = tf.reshape(x, [bsz, h * w, dim])
    if self.pos_encoding:
      x = x + tf.expand_dims(self.vis_pos_emb, 0)
    x = tf.concat([x, tf.expand_dims(self.emb_proj(t), 1)], 1)
    x, _ = self.transformer_decoder(x, c, None, None, None, training)
    x = x[:, :-1]  # remove time embedding.
    x = tf.reshape(x, [bsz, h, w, dim])
    x = self.trans_conv_out(self.trans_conv_out_norm(x))
    if self.strides > 1:
      x = tf.nn.depth_to_space(x, self.strides)
    return x


class TransUNet(tf.keras.layers.Layer):
  """TransUNet architecture."""

  def __init__(self,
               out_dim=3,
               dim=256,
               in_strides=1,
               in_kernel_size=3,
               out_kernel_size=3,
               n_res_blocks=(4, 4, 4),
               ch_multipliers=(1, 1, 1),
               kernel_sizes=(3, 3, 3),
               n_mlp_blocks=0,
               dropout=0.2,
               mhsa_resolutions=(16, 8),
               per_head_dim=64,
               transformer_dim=256,
               transformer_strides=2,
               transformer_blocks=4,
               pos_encoding='learned',
               conditioning=False,
               time_scaling=1e4,
               norm_type='group_norm',
               outp_softmax_groups=0,
               b_scale=1.,
               **kwargs):
    super().__init__(**kwargs)
    if isinstance(kernel_sizes, int):
      kernel_sizes = [kernel_sizes] * len(n_res_blocks)
    self.config = ml_collections.ConfigDict(
        initial_dictionary={
            'out_dim': out_dim,
            'dim': dim,
            'in_strides': in_strides,
            'in_kernel_size': in_kernel_size,
            'out_kernel_size': out_kernel_size,
            'n_res_blocks': n_res_blocks,
            'kernel_sizes': kernel_sizes,
            'ch_multipliers': ch_multipliers,
            'n_mlp_blocks': n_mlp_blocks,
            'dropout': dropout,
            'mhsa_resolutions': mhsa_resolutions,
            'per_head_dim': per_head_dim,
            'transformer_dim': transformer_dim,
            'transformer_strides': transformer_strides,
            'transformer_blocks': transformer_blocks,
            'pos_encoding': pos_encoding,
            'conditioning': conditioning,
            'time_scaling': time_scaling,
            'norm_type': norm_type,
            'outp_softmax_groups': outp_softmax_groups,
            'b_scale': b_scale,
        })

  def build(self, input_shape):
    config = self.config
    self.in_dim = input_shape[-1]
    self.time_emb = ScalarEmbedding(
        dim=config.dim,
        scaling=config.time_scaling,
        name='time_emb')

    self.conv_in = tf.keras.layers.Conv2D(
        filters=config.dim * config.ch_multipliers[0],
        kernel_size=[config.in_kernel_size, config.in_kernel_size],
        strides=[config.in_strides, config.in_strides],
        padding='SAME',
        use_bias=True,
        kernel_initializer=get_variable_initializer(1.0),
        name='conv_in')
    self.group_norm_in = get_norm(
        'group_norm',
        num_groups=min(config.dim * config.ch_multipliers[0] // 4, 32),
        name='group_norm_in')

    # dblocks and ublocks
    self.up_units = {}
    self.dn_units = {}
    for i, (n_res_blocks, kernel_size, ch_multiplier) in enumerate(zip(
        config.n_res_blocks,
        config.kernel_sizes,
        config.ch_multipliers)):
      self.dn_units[str(i)] = ConvUnit(
          config,
          n_res_blocks=n_res_blocks,
          kernel_size=kernel_size,
          ch_multiplier=ch_multiplier,
          expect_lateral=False,
          up_dn_sample='dn' if i != len(config.n_res_blocks) - 1 else 'none',
          name=f'dn_unit_{i}')
      self.up_units[str(i)] = ConvUnit(
          config,
          n_res_blocks=n_res_blocks + 1,
          kernel_size=kernel_size,
          ch_multiplier=ch_multiplier,
          expect_lateral=True,
          up_dn_sample='up' if i != 0 else 'none',
          name=f'up_unit_{i}')

    self.up_mlp_units = {}
    self.dn_mlp_units = {}
    for i in range(config.n_mlp_blocks):
      self.dn_mlp_units[str(i)] = ConvUnit(
          config,
          n_res_blocks=config.n_mlp_blocks,
          kernel_size=1,
          ch_multiplier=1,
          expect_lateral=False,
          up_dn_sample='none',
          name=f'dn_mlp_unit_{i}')
      self.up_mlp_units[str(i)] = ConvUnit(
          config,
          n_res_blocks=config.n_mlp_blocks,
          kernel_size=1,
          ch_multiplier=1,
          expect_lateral=True,
          up_dn_sample='none',
          name=f'up_mlp_unit_{i}')

    if config.transformer_blocks > 0:
      self.transformer = Transformer(
          dim=config.transformer_dim,
          strides=config.transformer_strides,
          num_layers=config.transformer_blocks,
          num_heads=config.transformer_dim // config.per_head_dim,
          mlp_ratio=4,
          drop_units=config.dropout,
          drop_att=0.,
          drop_path=0.,
          pos_encoding=config.pos_encoding,
          conditioning=config.conditioning,
          name='transformer')

    self.group_norm_out = get_norm(
        'group_norm',
        num_groups=min(config.dim // 4, 32),
        name='group_norm_out')
    if config.outp_softmax_groups == 0:
      self.conv_out = tf.keras.layers.Conv2D(
          filters=config.out_dim * config.in_strides**2,
          kernel_size=[config.out_kernel_size, config.out_kernel_size],
          strides=[1, 1],
          padding='SAME',
          use_bias=True,
          kernel_initializer=get_variable_initializer(),
          name='conv_out')
    else:
      nbits = config.out_dim // config.outp_softmax_groups
      self.conv_out = tf.keras.layers.Conv2D(
          filters=2**nbits * config.outp_softmax_groups,
          kernel_size=[1, 1],
          strides=[1, 1],
          padding='SAME',
          use_bias=True,
          kernel_initializer=get_variable_initializer(),
          name='conv_out')
      self.bits = int2bits(tf.range(2**nbits), nbits, tf.float32)

  def call(self, x, t, cembs, training, return_logits=False):
    """x in (bsz, h, w, c), t in (bsz,) or (bsz, k), cembs in (bsz, s, d)."""
    config = self.config
    emb = self.time_emb(t)
    x = self.group_norm_in(self.conv_in(x))

    x_list = [x]
    for i in range(config.n_mlp_blocks):
      x = self.dn_mlp_units[str(i)](x, emb, cembs, x_list, training)
    for i in range(len(config.n_res_blocks)):
      x = self.dn_units[str(i)](x, emb, cembs, x_list, training)
    if config.transformer_blocks > 0:
      x = self.transformer(x, emb, cembs, training)
    # for x_ in x_list + [x]:
    #   print('x shape', x_.shape)
    for i in range(len(config.n_res_blocks))[::-1]:
      x = self.up_units[str(i)](x, emb, cembs, x_list, training)
    for i in range(config.n_mlp_blocks)[::-1]:
      x = self.up_mlp_units[str(i)](x, emb, cembs, x_list, training)
    assert not x_list, 'x_list should be fully utilized'

    x = self.conv_out(tf.nn.silu(self.group_norm_out(x)))
    if config.in_strides > 1:
      x = tf.nn.depth_to_space(x, config.in_strides)

    if config.outp_softmax_groups > 0:
      xs_ori = xs = tf.split(x, config.outp_softmax_groups, -1)
      xs = [tf.einsum('bhwd,do->bhwo', tf.nn.softmax(x), self.bits) for x in xs]
      x = (tf.concat(xs, -1) * 2 - 1) * config.b_scale
      if return_logits:
        x = tf.stack(xs_ori, -2)  # (bsz, h, w, groups, n_class_per_group)

    return x
