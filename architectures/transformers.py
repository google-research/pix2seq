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
"""Transformer."""

import math

from architectures import resnet
import tensorflow as tf


def suffix_id(i):
  """Return suffix id for layer/variable name."""
  return '' if i == 0 else '_%d' % i


def get_shape(x):
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def get_angles(pos, i, dim):
  angle_rates = 1 / tf.pow(10000., tf.cast(2 * (i//2), tf.float32) / dim)
  return tf.cast(pos, tf.float32) * tf.cast(angle_rates, tf.float32)


def positional_encoding(coords, dim):
  """coords in (bsz, size), return (bsz, size, dim)."""
  angle_rads = get_angles(tf.expand_dims(coords, -1),
                          tf.range(dim)[tf.newaxis, tf.newaxis, :],
                          dim)

  # apply sin to even indices in the array; 2i
  angle_rads1 = tf.sin(angle_rads[:, :, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads2 = tf.cos(angle_rads[:, :, 1::2])

  pos_encoding = tf.concat([angle_rads1, angle_rads2], -1)

  return tf.cast(pos_encoding, dtype=tf.float32)


def get_1d_position_codes(seqlen, out_dim, normalization_max=6.2831852):
  """Get 2d positional embedding with sin/cos codes.

  Args:
    seqlen: a `int` specifying the length of the sequence.
    out_dim: a `int` specifying the output dimension of the encoding.
    normalization_max: normalize coordinates between [0, normalization_max].
      If None, raw coordinates from 0 to seqlen will be used.

  Returns:
    positional code of shape (1, seqlen, out_dim)
  """
  coords = tf.cast(tf.range(seqlen), tf.float32)
  if normalization_max is not None:
    coords = coords / (seqlen - 1) * normalization_max
  coords = positional_encoding(coords, out_dim)
  return coords


def get_2d_position_codes(height, width, out_dim, normalization_max=6.2831852):
  """Get 2d positional embedding with sin/cos codes.

  Args:
    height: a `int` specifying the height of the 2d image / feature map.
    width: a `int` specifying the width of the 2d image / feature map.
    out_dim: a `int` specifying the output dimension of the encoding.
      Must be divisible by 2.
    normalization_max: normalize coordinates between [0, normalization_max].
      If None, raw coordinates from 0 to height/width will be used.

  Returns:
    positional code of shape (1, height, width, out_dim)
  """
  y_coords = tf.cast(tf.range(height), tf.float32)
  if normalization_max is not None:
    y_coords = y_coords / (height - 1) * normalization_max
  y_coords = positional_encoding(y_coords, out_dim//2)
  y_coords = tf.expand_dims(y_coords, 2)
  y_coords = tf.concat([y_coords, tf.zeros_like(y_coords)], -1)

  x_coords = tf.cast(tf.range(width), tf.float32)
  if normalization_max is not None:
    x_coords = x_coords / (width - 1) * normalization_max
  x_coords = positional_encoding(x_coords, out_dim//2)
  x_coords = tf.expand_dims(x_coords, 1)
  x_coords = tf.concat([tf.zeros_like(x_coords), x_coords], -1)

  return y_coords + x_coords


def get_variable_initializer(name=None):
  if name is None:
    return tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)


def add_seq_pos_emb(self, pos_encoding, max_seq_len, dim,
                    name_prefix=None, initializer=None):
  """Add seq_pos_emb variable/tensor to model instance referenced by `self`."""
  if name_prefix is None:
    name_prefix = self.name
  if initializer is None:
    initializer = get_variable_initializer()
  if pos_encoding == 'learned':
    self.seq_pos_emb = self.add_weight(
        shape=(max_seq_len, dim), initializer=initializer,
        name='%s/seq_pos_embedding' % name_prefix)
  elif pos_encoding == 'sin_cos':
    sin_cos = get_1d_position_codes(
        max_seq_len, dim, normalization_max=6.2831852)
    self.seq_pos_emb = tf.reshape(sin_cos, [max_seq_len, dim])
  else:
    raise ValueError('Unknown pos encoding %s' % pos_encoding)


def add_vis_pos_emb(self, pos_encoding, n_rows, n_cols, dim,
                    name_prefix=None, initializer=None):
  """Add vis_pos_emb variable/tensor to model instance referenced by `self`."""
  if name_prefix is None:
    name_prefix = self.name
  if initializer is None:
    initializer = get_variable_initializer()
  if pos_encoding == 'learned':
    self.vis_pos_emb = self.add_weight(
        shape=(n_rows * n_cols, dim), initializer=initializer,
        name='%s/vis_pos_embedding' % name_prefix)
  elif pos_encoding == 'sin_cos':
    sin_cos = get_2d_position_codes(
        n_rows, n_cols, dim, normalization_max=6.2831852)
    self.vis_pos_emb = tf.reshape(sin_cos, [n_rows * n_cols, dim])
  else:
    raise ValueError('Unknown pos encoding %s' % pos_encoding)


def add_cls_token_emb(self, dim, name_prefix=None, initializer=None):
  """Add cls_token_emb variable to model instance referenced by `self`."""
  if name_prefix is None:
    name_prefix = self.name
  if initializer is None:
    initializer = get_variable_initializer()
  self.cls_token_emb = self.add_weight(
      shape=(1, dim), initializer=initializer,
      name='%s/cls_token_embedding' % name_prefix)


def add_vocab_token_emb(self, vocab_size, dim, shared_embedding, output_bias,
                        name_prefix=None, initializer=None):
  """Add token_embedding variable to model instance referenced by `self`."""
  if name_prefix is None:
    name_prefix = self.name
  if initializer is None:
    initializer = get_variable_initializer()
  if shared_embedding:
    self.token_embedding = self.add_weight(
        shape=[vocab_size, dim],
        initializer=initializer,
        name='%s/token_embedding' % name_prefix)
  else:
    self.inp_token_embedding = self.add_weight(
        shape=[vocab_size, dim],
        initializer=initializer,
        name='%s/inp_token_embedding' % name_prefix)
    self.outp_token_embedding = self.add_weight(
        shape=[vocab_size, dim],
        initializer=initializer,
        name='%s/outp_token_embedding' % name_prefix)
  if output_bias:
    self.outp_bias = self.add_weight(
        shape=[vocab_size],
        initializer=initializer,
        name='%s/outp_bias' % name_prefix)


def kronecker_product(mat1, mat2):
  """Computes the Kronecker product two matrices."""
  # m1, n1 = mat1.get_shape().as_list()
  sh1 = tf.shape(mat1)
  m1, n1 = sh1[0], sh1[1]
  mat1_rsh = tf.reshape(mat1, [m1, 1, n1, 1])
  # m2, n2 = mat2.get_shape().as_list()
  sh2 = tf.shape(mat2)
  m2, n2 = sh2[0], sh2[1]
  mat2_rsh = tf.reshape(mat2, [1, m2, 1, n2])
  return tf.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])


def get_ar_mask(seq_len, dtype=tf.float32):
  """Get autoregressive causal mask so the model cannot attends to the future.

  Args:
    seq_len: a `int` or `int` tensor specifying the sequence length.
    dtype: tf data type for the return tensor.

  Returns:
    tensor of shape [1, 1, seq_len, seq_len] with ones for locations to be
    masked out.
  """
  valid_locs = tf.linalg.band_part(
      tf.ones([seq_len, seq_len], dtype=dtype), -1, 0)
  valid_locs = tf.reshape(valid_locs, [1, 1, seq_len, seq_len])
  return 1.0 - valid_locs


def get_local_ar_mask(seq_len, window_size, dtype=tf.float32):
  """Get local causal mask of window size.

  The attention is causal within each window, but cannot exceed local window.

  Args:
    seq_len: a `int` or `int` tensor specifying the sequence length.
    window_size: a `int` or `int` tensor specifying the local window size.
      seq_len must be divisible by window_size.
    dtype: tf data type for the return tensor.

  Returns:
    tensor of shape [1, 1, seq_len, seq_len] with ones for
    locations to be masked out.
  """
  valid_locs = tf.linalg.band_part(
      tf.ones([window_size, window_size], dtype=dtype), -1, 0)
  valid_locs = kronecker_product(tf.eye(seq_len // window_size), valid_locs)
  valid_locs = tf.reshape(valid_locs, [1, 1, seq_len, seq_len])
  return 1.0 - valid_locs


def merge_masks(mask1, mask2):
  """Merge ar and local ar masks, each of shape (1, 1, src, dst)."""
  sh1 = tf.shape(mask1)
  sh2 = tf.shape(mask2)
  top_right = tf.ones([1, 1, sh1[2], sh2[3]], mask1.dtype)
  bottom_left = tf.zeros([1, 1, sh2[2], sh1[3]], mask2.dtype)
  return tf.concat([
      tf.concat([mask1, top_right], 3),
      tf.concat([bottom_left, mask2], 3)], 2)


def top_logits(logits: tf.Tensor,
               k: int = 0,
               p: float = 1.0,
               mask: float = -1e10) -> tf.Tensor:
  """Remove low probability logits via masking.

  Args:
    logits: class logits in shape of (batch size, total_classes).
    k: specifying top k largest logits to keep.
    p: specifying a probability for finding a minimum set of largest
      logits to keep, where their cumulative probability is no less than p
      (actually in the following version, it is "...cumulative probability is
      the largest but no more than p").
    mask: an value that's used to replace logits that don't satisfy the
      keep conditions.

  Returns:
    logits where low probability ones are replaced with mask.
  """
  mask = tf.ones_like(logits) * mask
  if k > 0:
    min_logits = tf.nn.top_k(logits, k=k)[0][:, -1:]
    logits = tf.where(logits < min_logits, mask, logits)
  if p < 1.:
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cum_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    min_logits = -tf.reduce_max(
        tf.where(cum_probs <= p, -sorted_logits, mask), -1, keepdims=True)
    min_logits = tf.minimum(min_logits, sorted_logits[:, :1])
    logits = tf.where(logits < min_logits, mask, logits)
  return logits


def sample_categorical(logits, num_classes, temperature, top_k, top_p):
  """logits in (..., num_classes), and return (...)."""
  out_shape = tf.shape(logits)[:-1]
  logits = tf.reshape(logits, [-1, num_classes])
  logits = logits  / tf.cast(temperature, tf.float32)
  logits = top_logits(logits, k=top_k, p=top_p)
  samples = tf.random.categorical(
      logits, num_samples=1, dtype=tf.int32)[:, 0]
  return tf.reshape(samples, out_shape)


def unfold(images, patch_size, patch_stride=None):
  if patch_stride is None:
    patch_stride = patch_size
  patches = tf.image.extract_patches(
      images,
      sizes=[1, patch_size, patch_size, 1],
      strides=[1, patch_stride, patch_stride, 1],
      rates=[1, 1, 1, 1],
      padding='VALID')
  return patches


class DropPath(tf.keras.layers.Layer):
  """For stochastic depth."""

  def __init__(self, drop_rate=0., **kwargs):
    """Initializes a drop path layer."""
    super(DropPath, self).__init__(**kwargs)
    self._drop_rate = drop_rate
    if self._drop_rate < 0 or self._drop_rate >= 1.0:
      raise ValueError('drop_rate {} is outside [0, 1)'.format(self._drop_rate))

  def call(self, x, training=False):
    """Performs a forward pass.

    Args:
      x: An input tensor of type tf.Tensor with shape [batch, height,
        width, channels].
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      The output tensor.
    """
    if self._drop_rate == 0. or not training:
      return x

    keep_rate = 1. - self._drop_rate
    xshape = tf.shape(x)
    drop_mask_shape = [xshape[0]] + [1] * (len(xshape) - 1)
    drop_mask = keep_rate + tf.random.uniform(drop_mask_shape, dtype=x.dtype)
    drop_mask = tf.math.divide(tf.floor(drop_mask), keep_rate)

    return x * drop_mask


class FeedForwardLayer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self, dim_att, dim_mlp, drop_units=0.1, **kwargs):
    super(FeedForwardLayer, self).__init__(**kwargs)
    self.dense1 = tf.keras.layers.Dense(
        dim_mlp, activation=tf.nn.gelu, name='dense1')
    self.dropout = tf.keras.layers.Dropout(drop_units)
    self.dense2 = tf.keras.layers.Dense(dim_att, name='dense2')

  def call(self, x, training):
    return self.dense2(self.dropout(self.dense1(x), training=training))


class MLP(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               num_layers,
               dim,
               mlp_ratio,
               drop_path=0.1,
               drop_units=0.,
               **kwargs):
    super(MLP, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.mlp_layers = [
        FeedForwardLayer(dim, dim * mlp_ratio, drop_units,
                         name='ffn' + suffix_id(i))
        for i in range(num_layers)
    ]
    self.layernorms = [
        tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='ffn/ln' + suffix_id(i))
        for i in range(num_layers)
    ]
    self.dropp = DropPath(drop_path)

  def call(self, x, training, ret_list=False):
    x_list = [x]
    for i in range(self.num_layers):
      x_residual = self.mlp_layers[i](self.layernorms[i](x), training)
      x = x + self.dropp(x_residual, training)
      x_list.append(x)
    return (x, x_list) if ret_list else x


class TransformerEncoderLayer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               dim,
               mlp_ratio,
               num_heads,
               drop_path=0.1,
               drop_units=0.1,
               drop_att=0.,
               **kwargs):
    super(TransformerEncoderLayer, self).__init__(**kwargs)
    self.mha_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='mha/ln')
    self.mha = tf.keras.layers.MultiHeadAttention(
        num_heads, dim // num_heads, dropout=drop_att, name='mha')
    self.mlp = MLP(1, dim, mlp_ratio, drop_path, drop_units, name='mlp')
    self.dropp = DropPath(drop_path)

  def call(self, x, mask, training):
    # x shape (bsz, seq_len, dim_att), mask shape (bsz, seq_len, seq_len).
    x_ln = self.mha_ln(x)
    x_residual = self.mha(x_ln, x_ln, x_ln, mask, training=training)
    x = x + self.dropp(x_residual, training)
    x = self.mlp(x, training)
    return x


class TransformerEncoder(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               num_layers,
               dim,
               mlp_ratio,
               num_heads,
               drop_path=0.1,
               drop_units=0.1,
               drop_att=0.,
               **kwargs):
    super(TransformerEncoder, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.enc_layers = [
        TransformerEncoderLayer(  # pylint: disable=g-complex-comprehension
            dim, mlp_ratio, num_heads, drop_path, drop_units, drop_att,
            name='transformer_encoder' + suffix_id(i))
        for i in range(num_layers)
    ]

  def call(self, x, mask, training, ret_list=False):
    x_list = [x]
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, mask, training)
      x_list.append(x)
    return (x, x_list) if ret_list else x


class TransformerDecoderLayer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               dim,
               mlp_ratio,
               num_heads,
               drop_path=0.1,
               drop_units=0.1,
               drop_att=0.,
               self_attention=True,
               cross_attention=True,
               **kwargs):
    super(TransformerDecoderLayer, self).__init__(**kwargs)
    self.self_attention = self_attention
    self.cross_attention = cross_attention
    if self_attention:
      self.self_ln = tf.keras.layers.LayerNormalization(
          epsilon=1e-6, name='self_mha/ln')
      self.self_mha = tf.keras.layers.MultiHeadAttention(
          num_heads, dim // num_heads, dropout=drop_att, name='self_mha')
    if cross_attention:
      self.cross_ln = tf.keras.layers.LayerNormalization(
          epsilon=1e-6, name='cross_mha/ln')
      self.cross_mha = tf.keras.layers.MultiHeadAttention(
          num_heads, dim // num_heads, dropout=drop_att, name='cross_mha')
    self.mlp = MLP(1, dim, mlp_ratio, drop_path, drop_units, name='mlp')
    self.dropp = DropPath(drop_path)

  def call(self, x, enc, cache, mask_self, mask_cross, training):
    """x in (bsz, seq, d), enc in (bsz, seq', d)."""
    x_for_cache = []
    if self.self_attention:
      x_for_cache = x_ln = kv_ln = self.self_ln(x)
      if cache is not None:  # Augment kv_ln with cache in (bsz, c_size, d).
        q_size, k_size = tf.shape(x)[1], tf.shape(cache)[1]
        mask_self = tf.concat([tf.ones([1, 1, q_size, k_size]), mask_self], -1)
        kv_ln = tf.concat([cache, x_ln], axis=1)
      x_res = self.self_mha(x_ln, kv_ln, kv_ln, mask_self, training=training)
      x = x + self.dropp(x_res, training)
    if self.cross_attention:
      x_ln = self.cross_ln(x)
      x_res = self.cross_mha(x_ln, enc, enc, mask_cross, training=training)
      x = x + self.dropp(x_res, training)
    x = self.mlp(x, training)
    return x, x_for_cache


class TransformerDecoder(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               num_layers,
               dim,
               mlp_ratio,
               num_heads,
               drop_path=0.1,
               drop_units=0.1,
               drop_att=0.,
               self_attention=True,
               cross_attention=True,
               **kwargs):
    super(TransformerDecoder, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.dec_layers = [
        TransformerDecoderLayer(  # pylint: disable=g-complex-comprehension
            dim, mlp_ratio, num_heads, drop_path, drop_units, drop_att,
            self_attention, cross_attention,
            name='transformer_decoder_layer' + suffix_id(i))
        for i in range(num_layers)
    ]

  def call(self, x, enc, caches, mask_self, mask_cross, training):
    """x in (bsz, seq, d), enc in (bsz, seq', d)."""
    presents = []
    for i in range(self.num_layers):
      cache = None if caches is None else caches[i]
      x, x_for_cache = self.dec_layers[i](
          x, enc, cache, mask_self, mask_cross, training)
      presents.append(x_for_cache)

    return x, tf.stack(presents)


class VisionTransformer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               image_height,
               image_width,
               patch_size,
               num_layers,
               dim,
               mlp_ratio,
               num_heads,
               drop_path=0.1,
               drop_units=0.1,
               drop_att=0.,
               pos_encoding='learned',
               use_cls_token=True,
               **kwargs):
    super(VisionTransformer, self).__init__(**kwargs)
    self.use_cls_token = use_cls_token
    self.patch_size = patch_size
    self.stem_conv = tf.keras.layers.Conv2D(
        filters=dim, kernel_size=patch_size, strides=patch_size,
        padding='VALID', use_bias=True, name='stem_conv')
    self.stem_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='stem_ln')
    if self.use_cls_token:
      add_cls_token_emb(self, dim)
    self.n_rows, self.n_cols = image_height//patch_size, image_width//patch_size
    add_vis_pos_emb(self, pos_encoding, self.n_rows, self.n_cols, dim)
    self.transformer_encoder = TransformerEncoder(
        num_layers, dim, mlp_ratio, num_heads, drop_path, drop_units, drop_att,
        name='transformer_encoder')
    self.output_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='ouput_ln')

  def call(self, images, training, ret_list=False):
    """Input images of (bsz, h, w, c)."""
    tokens = self.stem_conv(images)
    bsz, h, w, dim = get_shape(tokens)
    tokens = self.stem_ln(tf.reshape(tokens, [bsz, h * w, dim]))

    tokens = tokens + tf.expand_dims(self.vis_pos_emb, 0)
    if self.use_cls_token:
      cls_token = tf.tile(tf.expand_dims(self.cls_token_emb, 0), [bsz, 1, 1])
      tokens = tf.concat([cls_token, tokens], 1)

    tokens, x_list = self.transformer_encoder(
        tokens, None, training=training, ret_list=True)
    x = self.output_ln(tokens)
    return (x, x_list) if ret_list else x


class ResNetTransformer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               image_height,
               image_width,
               resnet_variant,
               resnet_depth,
               resnet_width_multiplier,
               resnet_sk_ratio,
               num_layers,
               dim,
               mlp_ratio,
               num_heads,
               drop_path=0.1,
               drop_units=0.1,
               drop_att=0.,
               pos_encoding='learned',
               use_cls_token=True,
               **kwargs):
    super(ResNetTransformer, self).__init__(**kwargs)
    self.use_cls_token = use_cls_token
    self.resnet = resnet.resnet(
        resnet_depth=resnet_depth,
        width_multiplier=resnet_width_multiplier,
        sk_ratio=resnet_sk_ratio,
        variant=resnet_variant)
    self.dropout = tf.keras.layers.Dropout(drop_units)
    self.stem_projection = tf.keras.layers.Dense(dim, name='stem_projection')
    self.stem_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='stem_ln')
    if self.use_cls_token:
      add_cls_token_emb(self, dim)
    if resnet_variant in ['c3']:
      factor = 8.
    elif resnet_variant in ['c4', 'dc5']:
      factor = 16.
    else:
      factor = 32.
    self.n_rows = math.ceil(image_height / factor)
    self.n_cols = math.ceil(image_width / factor)
    add_vis_pos_emb(self, pos_encoding, self.n_rows, self.n_cols, dim)
    self.transformer_encoder = TransformerEncoder(
        num_layers, dim, mlp_ratio, num_heads, drop_path, drop_units, drop_att,
        name='transformer_encoder')
    self.output_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='ouput_ln')

  def call(self, images, training, ret_list=False):
    """Input images of (bsz, h, w, c)."""
    hidden_stack, _ = self.resnet(images, training)
    tokens = hidden_stack[-1]
    bsz, h, w, num_channels = get_shape(tokens)
    tokens = tf.reshape(tokens, [bsz, h * w, num_channels])
    tokens = self.stem_ln(self.stem_projection(self.dropout(tokens, training)))

    tokens = tokens + tf.expand_dims(self.vis_pos_emb, 0)
    if self.use_cls_token:
      cls_token = tf.tile(tf.expand_dims(self.cls_token_emb, 0), [bsz, 1, 1])
      tokens = tf.concat([cls_token, tokens], 1)

    tokens, x_list = self.transformer_encoder(
        tokens, None, training=training, ret_list=True)
    x = self.output_ln(tokens)
    return (x, x_list) if ret_list else x


class AutoregressiveDecoder(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               vocab_size,
               max_seq_len,
               num_layers,
               dim,
               mlp_ratio,
               num_heads,
               drop_path=0.1,
               drop_units=0.1,
               drop_att=0.,
               pos_encoding='learned',
               shared_embedding=True,
               output_bias=True,
               **kwargs):
    super(AutoregressiveDecoder, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.max_seq_len = max_seq_len
    self.num_layers = num_layers
    self.dim = dim
    self.shared_embedding = shared_embedding
    self.output_bias = output_bias
    add_seq_pos_emb(self, pos_encoding, max_seq_len, dim)
    add_vocab_token_emb(self, vocab_size, dim, shared_embedding, output_bias)
    self.decoder = TransformerDecoder(
        num_layers, dim, mlp_ratio, num_heads,
        drop_path, drop_units, drop_att, name='transformer_decoder')
    self.output_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='ouput_ln')

  def call(self, tokens, encoded, training):
    """Teacher-forced prediction.

    Args:
      tokens: `int` tokens with shape of (bsz, seqlen, dim).
      encoded: `float` encoded representations for conditioning with shape of
        (bsz, size, dim). This can be optional in case of pure decoder.
      training: `boolean` indicator for training vs test mode.

    Returns:
      logits of `float` with shape of (bsz, seqlen, vocab_size)
    """
    _, seqlen = get_shape(tokens)
    seq_pos_emb = tf.expand_dims(self.seq_pos_emb[:seqlen], 0)
    if self.shared_embedding:
      inp_embedding = outp_embedding = self.token_embedding
    else:
      inp_embedding = self.inp_token_embedding
      outp_embedding = self.outp_token_embedding

    token_emb = tf.gather(inp_embedding, tokens) + seq_pos_emb
    mask_self = 1. - get_ar_mask(seqlen, token_emb.dtype)
    outputs, _ = self.decoder(
        token_emb, encoded, None, mask_self, None, training)
    outputs = self.output_ln(outputs)
    logits = tf.matmul(outputs, outp_embedding, transpose_b=True)
    if self.output_bias:
      logits = tf.nn.bias_add(logits, self.outp_bias)
    return logits

  def infer(self, prompt, encoded, max_seq_len=None,
            temperature=1.0, top_k=1, top_p=1.0, sampling_callback=None):
    """Autoregressive (without teacher-forcing) prediction.

    Note: the autoregressive sampling/inference time can be further optimized by
    caching *transformed* key / value inside multi-head attention for the
    `encoded` and previously generated tokens, but this may make the code less
    readable.

    Args:
      prompt: `int` tokens with shape of (bsz, prompt_len).
      encoded: `float` encoded representations for conditioning with shape of
        (bsz, size, dim). This can be optional in case of pure decoder.
      max_seq_len: `int` of max generated sequence length (including prompt).
      temperature: `float` scalar for scaling the logits before sampling.
      top_k: `int` scalar for truncating top-k tokens according to logits before
        token sampling.
      top_p: `float` scalar specifying the threshold of cumulative probablity
        for truncating tokens before token sampling.
      sampling_callback: a callbak `function` that take `next_logits`, and
        return `next_token`. This is used when users need a specific logic
        for sampling. Default to `None` with standard free-form sampling.

    Returns:
      sampled tokens with shape of (bsz, max_seq_len-prompt_len).
      logits (temperature-scaled) associated with sampled token, in shape of
        (bsz, max_seq_len-prompt_len, vocab_size).
    """
    bsz, prompt_len = get_shape(prompt)
    seq_len = self.max_seq_len if max_seq_len is None else max_seq_len
    seq_pos_emb = tf.expand_dims(self.seq_pos_emb, 0)
    if self.shared_embedding:
      inp_embedding = outp_embedding = self.token_embedding
    else:
      inp_embedding = self.inp_token_embedding
      outp_embedding = self.outp_token_embedding

    # Each step reads caches[:step] and tokens[step:next_step] and updates
    # tokens[next_step], logits[next_step] and caches[step:next_step].
    # On the first step, step=0, next_step=prompt_len. On subsequent steps
    # next_step = step + 1.
    def loop_body(step, caches, tokens, logits, is_prompt=False):
      if is_prompt:
        assert step == 0
        x = tf.gather(inp_embedding, tf.transpose(tokens[:prompt_len]))
        x = x + seq_pos_emb[:, :prompt_len]  # (bsz, prompt_len, d)
        mask_self = 1. - get_ar_mask(prompt_len, x.dtype)
        caches_in = None
      else:
        x = tf.gather(inp_embedding, tf.transpose(tokens[step]))
        x = x + seq_pos_emb[:, step]  # (bsz, d)
        x = tf.expand_dims(x, 1)  # (bsz, 1, d)
        mask_self = tf.ones([1, 1, 1, 1])
        caches_in = tf.transpose(caches[:step], [1, 2, 0, 3])
      outputs, caches_out = self.decoder(
          x, encoded, caches_in, mask_self, None, training=False)
      outputs = self.output_ln(outputs)
      next_logits = tf.matmul(  # only take the last for sampling next token.
          outputs, outp_embedding, transpose_b=True)[:, -1]
      if self.output_bias:
        next_logits = tf.nn.bias_add(next_logits, self.outp_bias)

      # Scale and trunctate logits and sample next token.
      if sampling_callback:
        next_token = sampling_callback(
            next_logits, step, temperature, top_k, top_p)
      else:
        sampling_logits = next_logits / tf.cast(temperature, tf.float32)
        sampling_logits = top_logits(sampling_logits, k=top_k, p=top_p)
        next_token = tf.random.categorical(
            sampling_logits, num_samples=1, dtype=tf.int32)[:, 0]

      # Update internal states.
      next_step = step + (prompt_len if is_prompt else 1)
      caches_out = tf.transpose(caches_out, [2, 0, 1, 3])
      # TODO(srbs): We could merge these two branches by using
      # tf.tensor_scatter_nd_update(caches, tf.range(start, next_ste), ...)
      # but tf.range is not supported on TPU. If we could directly
      # use XLA's DynamicUpdateSlice here this wouldn't be a problem but that
      # is not exposed via TF's API.
      if is_prompt:
        caches = tf.tensor_scatter_nd_update(
            caches,
            tf.constant(list(range(prompt_len)))[:, tf.newaxis], caches_out)
      else:
        caches = tf.tensor_scatter_nd_update(caches, [[step]], caches_out)
      tokens = tf.tensor_scatter_nd_update(tokens, [[next_step]], [next_token])
      logits = tf.tensor_scatter_nd_update(logits, [[next_step]], [next_logits])
      return (next_step, caches, tokens, logits)

    def cond(step, caches, tokens, logits):
      del caches
      del tokens
      del logits
      return tf.less(step, seq_len-1)

    caches_var = tf.zeros([seq_len-1, self.num_layers, bsz, self.dim])
    tokens_var = tf.zeros([seq_len, bsz], dtype=tf.int64)
    logits_var = tf.zeros([seq_len, bsz, self.vocab_size], dtype=tf.float32)
    indices = tf.expand_dims(tf.range(prompt_len), -1)
    tokens_var = tf.tensor_scatter_nd_update(
        tokens_var, indices, tf.transpose(prompt, [1, 0]))

    step = 0
    step, caches_var, tokens_var, logits_var = loop_body(
        step, caches_var, tokens_var, logits_var, is_prompt=True)
    if seq_len > prompt_len:
      step, caches_var, tokens_var, logits_var = tf.while_loop(
          cond=cond, body=loop_body,
          loop_vars=[step, caches_var, tokens_var, logits_var])

    sampled_tokens = tf.transpose(tokens_var[prompt_len:], [1, 0])
    sampled_tokens_logits = tf.transpose(logits_var[prompt_len:], [1, 0, 2])
    return sampled_tokens, sampled_tokens_logits
