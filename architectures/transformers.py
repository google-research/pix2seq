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
import re
import einops

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
    y_coords = (
        y_coords / tf.cast(height - 1, dtype=tf.float32) * normalization_max)
  y_coords = positional_encoding(y_coords, out_dim//2)
  y_coords = tf.expand_dims(y_coords, 2)
  y_coords = tf.concat([y_coords, tf.zeros_like(y_coords)], -1)

  x_coords = tf.cast(tf.range(width), tf.float32)
  if normalization_max is not None:
    x_coords = (
        x_coords / tf.cast(width - 1, dtype=tf.float32) * normalization_max)
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


def add_vis_pos_emb(self,
                    pos_encoding,
                    n_rows,
                    n_cols,
                    dim,
                    name_prefix=None,
                    initializer=None,
                    return_only=False,
                    normalization_max=6.2831852):
  """Add vis_pos_emb variable/tensor to model instance referenced by `self`."""
  if name_prefix is None:
    name_prefix = self.name
  if initializer is None:
    initializer = get_variable_initializer()
  if pos_encoding == 'learned':
    vis_pos_emb = self.add_weight(
        shape=(n_rows * n_cols, dim), initializer=initializer,
        name='%s/vis_pos_embedding' % name_prefix)
  elif pos_encoding == 'sin_cos':
    if n_rows == 1 or n_cols == 1:
      sin_cos = get_1d_position_codes(
          n_rows * n_cols, dim, normalization_max=normalization_max)
    else:
      sin_cos = get_2d_position_codes(
          n_rows, n_cols, dim, normalization_max=normalization_max)
    vis_pos_emb = tf.reshape(sin_cos, [n_rows * n_cols, dim])
  else:
    raise ValueError('Unknown pos encoding %s' % pos_encoding)
  if not return_only:
    self.vis_pos_emb = vis_pos_emb
  return vis_pos_emb


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


def get_chunk_ar_mask(seq_len, chunk_size, dtype=tf.float32):
  """Get causal mask across chuncks, but full attention within each chunk.

  Args:
    seq_len: a `int` or `int` tensor specifying the sequence length.
    chunk_size: a `int` or `int` tensor specifying the local window size.
      seq_len must be divisible by chunk_size.
    dtype: tf data type for the return tensor.

  Returns:
    tensor of shape [1, 1, seq_len, seq_len] with ones for
    locations to be masked out.
  """
  valid_locs = tf.ones([chunk_size, chunk_size], dtype=dtype)
  valid_locs = kronecker_product(tf.eye(seq_len // chunk_size), valid_locs)
  valid_locs = tf.reshape(valid_locs, [1, 1, seq_len, seq_len])

  return get_ar_mask(seq_len) * (1.0 - valid_locs)


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

  def __init__(self,
               dim_att,
               dim_mlp,
               drop_units=0.1,
               use_ln=False,
               ln_scale_shift=False,
               **kwargs):
    super(FeedForwardLayer, self).__init__(**kwargs)
    self.dense1 = tf.keras.layers.Dense(
        dim_mlp, activation=tf.nn.gelu, name='dense1')
    self.dropout = tf.keras.layers.Dropout(drop_units)
    self.dense2 = tf.keras.layers.Dense(dim_att, name='dense2')
    if use_ln:
      self.ln = tf.keras.layers.LayerNormalization(
          epsilon=1e-6,
          center=ln_scale_shift,
          scale=ln_scale_shift,
          name='mlp_ln')
    else:
      self.ln = lambda x: x

  def call(self, x, training):
    return self.dense2(self.dropout(self.ln(self.dense1(x)), training=training))


class MLP(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               num_layers,
               dim,
               mlp_ratio,
               drop_path=0.1,
               drop_units=0.,
               use_ffn_ln=False,
               ln_scale_shift=True,
               **kwargs):
    super(MLP, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.mlp_layers = [
        FeedForwardLayer(dim, dim * mlp_ratio, drop_units,
                         use_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift,
                         name='ffn' + suffix_id(i))
        for i in range(num_layers)
    ]
    self.layernorms = [
        tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            center=ln_scale_shift,
            scale=ln_scale_shift,
            name='ffn/ln' + suffix_id(i))
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
               self_attention=True,
               use_ffn_ln=False,
               ln_scale_shift=True,
               **kwargs):
    super(TransformerEncoderLayer, self).__init__(**kwargs)
    self.self_attention = self_attention
    if self_attention:
      self.mha_ln = tf.keras.layers.LayerNormalization(
          epsilon=1e-6,
          center=ln_scale_shift,
          scale=ln_scale_shift,
          name='mha/ln')
      self.mha = tf.keras.layers.MultiHeadAttention(
          num_heads, dim // num_heads, dropout=drop_att, name='mha')
    self.mlp = MLP(1, dim, mlp_ratio, drop_path, drop_units,
                   use_ffn_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift,
                   name='mlp')
    self.dropp = DropPath(drop_path)

  def call(self, x, mask, training):
    # x shape (bsz, seq_len, dim_att), mask shape (bsz, seq_len, seq_len).
    if self.self_attention:
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
               self_attention=True,
               use_ffn_ln=False,
               ln_scale_shift=True,
               **kwargs):
    super(TransformerEncoder, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.enc_layers = [
        TransformerEncoderLayer(  # pylint: disable=g-complex-comprehension
            dim,
            mlp_ratio,
            num_heads,
            drop_path,
            drop_units,
            drop_att,
            self_attention=self_attention,
            use_ffn_ln=use_ffn_ln,
            ln_scale_shift=ln_scale_shift,
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
               dim_x_att=None,
               self_attention=True,
               cross_attention=True,
               use_mlp=True,
               use_enc_ln=False,
               use_ffn_ln=False,
               ln_scale_shift=True,
               **kwargs):
    super(TransformerDecoderLayer, self).__init__(**kwargs)
    self.self_attention = self_attention
    self.cross_attention = cross_attention
    self.use_mlp = use_mlp
    if self_attention:
      self.self_ln = tf.keras.layers.LayerNormalization(
          epsilon=1e-6,
          center=ln_scale_shift,
          scale=ln_scale_shift,
          name='self_mha/ln')
      self.self_mha = tf.keras.layers.MultiHeadAttention(
          num_heads, dim // num_heads, dropout=drop_att, name='self_mha')
    if cross_attention:
      self.cross_ln = tf.keras.layers.LayerNormalization(
          epsilon=1e-6,
          center=ln_scale_shift,
          scale=ln_scale_shift,
          name='cross_mha/ln')
      if use_enc_ln:
        self.enc_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            center=ln_scale_shift,
            scale=ln_scale_shift,
            name='cross_mha/enc_ln')
      else:
        self.enc_ln = lambda x: x
      dim_x_att = dim if dim_x_att is None else dim_x_att
      self.cross_mha = tf.keras.layers.MultiHeadAttention(
          num_heads, dim_x_att // num_heads, dropout=drop_att, name='cross_mha')
    if use_mlp:
      self.mlp = MLP(1, dim, mlp_ratio, drop_path, drop_units,
                     use_ffn_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift,
                     name='mlp')
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
      enc = self.enc_ln(enc)
      x_res = self.cross_mha(x_ln, enc, enc, mask_cross, training=training)
      x = x + self.dropp(x_res, training)
    if self.use_mlp:
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
               dim_x_att=None,
               self_attention=True,
               cross_attention=True,
               use_mlp=True,
               use_enc_ln=False,
               use_ffn_ln=False,
               ln_scale_shift=True,
               **kwargs):
    super(TransformerDecoder, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.dec_layers = [
        TransformerDecoderLayer(  # pylint: disable=g-complex-comprehension
            dim,
            mlp_ratio,
            num_heads,
            drop_path,
            drop_units,
            drop_att,
            dim_x_att=dim_x_att,
            self_attention=self_attention,
            cross_attention=cross_attention,
            use_mlp=use_mlp,
            use_enc_ln=use_enc_ln,
            use_ffn_ln=use_ffn_ln,
            ln_scale_shift=ln_scale_shift,
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
    return (x, hidden_stack) if ret_list else x


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
               cross_attention=True,
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
        drop_path, drop_units, drop_att,
        cross_attention=cross_attention, name='transformer_decoder')
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


def get_layer_config(layers_str):
  layer_configs = []
  if layers_str.endswith(')'):
    layers_str += '0'
  if not re.match(r'^(\([0-9]+\)[0-9]+)+$', layers_str):
    raise ValueError(f'Unrecognized layer specification {layers_str}.')
  blocks = re.findall(r'\([0-9]+\)[0-9]+', layers_str)
  for i, block in enumerate(blocks):
    x_layers, l_layers = map(int, re.findall('([0-9]+)', block))
    if i < len(blocks) - 1:
      assert l_layers > 0
    layer_configs.append((x_layers, l_layers))
  return layer_configs


class FIT(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               layers,  # str format: (local-layers)global-layers.. eg '(2)4(2)'
               x_size,
               num_groups,
               latents_per_group,
               x_dim,
               latent_dim,
               x_num_heads,
               latent_num_heads,
               mlp_ratio=4,
               drop_path=0.,
               drop_units=0.,
               drop_att=0.,
               x_pos_encoding='learned',
               latent_pos_encoding='learned',
               mask='none',
               **kwargs):
    super().__init__(**kwargs)
    if x_size % num_groups != 0:
      raise ValueError(
          f'x_size={x_size} is not divisible by num_groups={num_groups}')
    x_per_group = x_size // num_groups
    self.num_groups = num_groups
    self.latents_per_group = latents_per_group
    self.layer_configs = get_layer_config(layers)
    self.mask = None
    if mask == 'causal':
      self.mask = 1. - get_chunk_ar_mask(
          num_groups * latents_per_group, latents_per_group, dtype=tf.float32)
    elif mask != 'none':
      raise ValueError(f'Unknown mask {mask}')

    self.stem = tf.keras.layers.Dense(x_dim, name='stem')
    self.stem_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='stem_ln')
    self.stem_y = tf.keras.layers.Dense(x_dim, name='stem_y')
    self.stem_y_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='stem_y_ln')
    self.latent_pos_emb = add_vis_pos_emb(
        self, latent_pos_encoding, num_groups, latents_per_group, latent_dim,
        name_prefix=f'{self.name}/latent_pos_emb/kernel',
        return_only=True, normalization_max=1000.)
    if x_pos_encoding == 'sin_cos2d':
      s_ = tf.cast(tf.math.sqrt(tf.cast(x_per_group, tf.float32)), tf.int32)
      self.x_pos_emb = add_vis_pos_emb(
          self, 'sin_cos', s_, s_, x_dim,
          name_prefix=f'{self.name}/x_pos_emb/kernel',
          return_only=True, normalization_max=1000.)
    else:
      self.x_pos_emb = add_vis_pos_emb(
          self, x_pos_encoding, x_per_group, 1, x_dim,
          name_prefix=f'{self.name}/x_pos_emb/kernel',
          return_only=True, normalization_max=1000.)

    self.x2l_cross_attn = {}
    self.l2x_cross_attn = {}
    self.x_network = {}
    self.l_network = {}
    for i, (x_layers, l_layers) in enumerate(self.layer_configs):
      self.x_network[str(i)] = TransformerEncoder(
          x_layers,
          x_dim,
          mlp_ratio,
          x_num_heads,
          drop_path,
          drop_units,
          drop_att,
          name='x_network' + suffix_id(i))
      if l_layers > 0:
        self.l2x_cross_attn[str(i)] = TransformerDecoderLayer(
            dim=0,
            mlp_ratio=0,
            num_heads=min(latent_num_heads, x_num_heads),
            drop_path=0.,
            drop_units=0.,
            drop_att=0.,
            dim_x_att=min(latent_dim, x_dim),
            self_attention=False,
            cross_attention=True,
            use_mlp=False,
            use_enc_ln=True,
            name='l2x_cross_attn' + suffix_id(i))
        self.l_network[str(i)] = TransformerEncoder(
            l_layers,
            dim=latent_dim,
            mlp_ratio=mlp_ratio,
            num_heads=latent_num_heads,
            drop_path=drop_path,
            drop_units=drop_units,
            drop_att=drop_att,
            name='l_network' + suffix_id(i))
      if i < len(self.layer_configs) - 1:
        self.x2l_cross_attn[str(i)] = TransformerDecoderLayer(
            dim=0,
            mlp_ratio=0,
            num_heads=min(x_num_heads, latent_num_heads),
            drop_path=0.,
            drop_units=0.,
            drop_att=0.,
            dim_x_att=min(latent_dim, x_dim),
            self_attention=False,
            cross_attention=True,
            use_mlp=False,
            use_enc_ln=True,
            name='x2l_cross_attn' + suffix_id(i))
    self.x_output_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='x_output_ln')
    self.l_output_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='l_output_ln')

  def call(self, x, y=None, training=True):
    """x in [bsz, t, n, c], y in [bsz, t, m, k]."""
    bsz = tf.shape(x)[0]
    t, latents_per_group = self.num_groups, self.latents_per_group
    x = self.stem_ln(self.stem(x))
    x = x + self.x_pos_emb[tf.newaxis, tf.newaxis, ...]
    if y is not None:
      x = tf.concat([self.stem_y_ln(self.stem_y(y)), x], 2)
    latents = tf.reshape(self.latent_pos_emb, [1, t, latents_per_group, -1])
    latents = tf.tile(latents, [bsz, 1, 1, 1])

    latents = einops.rearrange(latents, 'b t m d -> (b t) m d', t=t)
    x = einops.rearrange(x, 'b t n c -> (b t) n c')
    for i in range(len(self.layer_configs)):
      x = self.x_network[str(i)](x, None, training=training)

      if self.layer_configs[i][-1] > 0:
        latents = self.l2x_cross_attn[str(i)](
            latents, x, None, None, None, training=training)[0]
        latents = einops.rearrange(latents, '(b t) m d -> b (t m) d', t=t)
        latents = self.l_network[str(i)](latents, self.mask, training)
        latents = einops.rearrange(latents, 'b (t m) d -> (b t) m d', t=t)
        if i < len(self.layer_configs) - 1:
          x = self.x2l_cross_attn[str(i)](
              x, latents, None, None, None, training=training)[0]

    x = einops.rearrange(x, '(b t) m d -> b t m d', t=t)
    latents = einops.rearrange(latents, '(b t) m d -> b t m d', t=t)
    return self.x_output_ln(x), self.l_output_ln(latents)


class FITDenoiser(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               layers,  # str format: (local-layers)global-layers.. eg '(2)4(2)'
               x_size,
               num_groups,
               latents_per_group,
               x_dim,
               latent_dim,
               x_num_heads,
               latent_num_heads,
               out_dim,
               mlp_ratio=4,
               drop_path=0.,
               drop_units=0.,
               drop_att=0.,
               x_pos_encoding='learned',
               latent_pos_encoding='learned',
               mask='none',
               cond_proj=True,
               self_cond='none',
               x_self_attention=True,
               xattn_with_mlp=False,
               xattn_enc_ln=True,
               **kwargs):
    super().__init__(**kwargs)
    if x_size % num_groups != 0:
      raise ValueError(
          f'x_size={x_size} is not divisible by num_groups={num_groups}')
    x_per_group = x_size // num_groups
    self.num_groups = num_groups
    self.latents_per_group = latents_per_group
    self.latent_dim = latent_dim
    self.self_cond = self_cond
    self.layer_configs = get_layer_config(layers)
    self.mask = None
    if mask == 'causal':
      self.mask = 1. - get_chunk_ar_mask(
          num_groups * latents_per_group, latents_per_group, dtype=tf.float32)
    elif mask != 'none':
      raise ValueError(f'Unknown mask {mask}')

    self.stem = tf.keras.layers.Dense(x_dim, name='stem')
    self.stem_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='stem_ln')
    self.time_emb = ScalarEmbedding(
        dim=latent_dim // 4,
        scaling=1000.,
        expansion=4,
        name='time_emb')
    if cond_proj:
      self.cond_proj = tf.keras.layers.Dense(latent_dim, name='cond_proj')
    else:
      self.cond_proj = lambda x: x
    self.latent_pos_emb = add_vis_pos_emb(
        self, latent_pos_encoding, num_groups, latents_per_group, latent_dim,
        name_prefix=f'{self.name}/latent_pos_emb/kernel',
        return_only=True, normalization_max=1000.)
    self.x_pos_emb = add_vis_pos_emb(
        self, x_pos_encoding, x_per_group, 1, x_dim,
        name_prefix=f'{self.name}/x_pos_emb/kernel',
        return_only=True, normalization_max=1000.)

    if self_cond == 'latent':
      self.latent_prev_proj = MLP(
          num_layers=1,
          dim=latent_dim,
          mlp_ratio=mlp_ratio,
          drop_path=0.,
          drop_units=0.,
          name='latent_prev_proj')
      self.latent_prev_ln = tf.keras.layers.LayerNormalization(
          epsilon=1e-6, gamma_initializer='zeros', name='latent_prev_ln')

    self.x2l_cross_attn = {}
    self.l2x_cross_attn = {}
    self.l2c_cross_attn = {}
    self.x_network = {}
    self.l_network = {}
    for i, (x_layers, l_layers) in enumerate(self.layer_configs):
      self.x_network[str(i)] = TransformerEncoder(
          x_layers,
          x_dim,
          mlp_ratio,
          x_num_heads,
          drop_path,
          drop_units,
          drop_att,
          self_attention=x_self_attention,
          name='x_network' + suffix_id(i))
      if l_layers > 0:
        self.l2x_cross_attn[str(i)] = TransformerDecoderLayer(
            dim=latent_dim,
            mlp_ratio=mlp_ratio,
            num_heads=min(latent_num_heads, x_num_heads),
            drop_path=0.,
            drop_units=0.,
            drop_att=0.,
            dim_x_att=min(latent_dim, x_dim),
            self_attention=False,
            cross_attention=True,
            use_mlp=xattn_with_mlp,
            use_enc_ln=xattn_enc_ln,
            name='l2x_cross_attn' + suffix_id(i))
        self.l2c_cross_attn[str(i)] = TransformerDecoderLayer(
            dim=0,
            mlp_ratio=0,
            num_heads=latent_num_heads,
            drop_path=0.,
            drop_units=0.,
            drop_att=0.,
            dim_x_att=latent_dim,
            self_attention=False,
            cross_attention=True,
            use_mlp=False,
            use_enc_ln=xattn_enc_ln,
            name='l2c_cross_attn' + suffix_id(i))
        self.l_network[str(i)] = TransformerEncoder(
            l_layers,
            dim=latent_dim,
            mlp_ratio=mlp_ratio,
            num_heads=latent_num_heads,
            drop_path=drop_path,
            drop_units=drop_units,
            drop_att=drop_att,
            name='l_network' + suffix_id(i))
      if i < len(self.layer_configs) - 1:
        self.x2l_cross_attn[str(i)] = TransformerDecoderLayer(
            dim=x_dim,
            mlp_ratio=mlp_ratio,
            num_heads=min(x_num_heads, latent_num_heads),
            drop_path=0.,
            drop_units=0.,
            drop_att=0.,
            dim_x_att=min(latent_dim, x_dim),
            self_attention=False,
            cross_attention=True,
            use_mlp=xattn_with_mlp,
            use_enc_ln=xattn_enc_ln,
            name='x2l_cross_attn' + suffix_id(i))
    self.x_output_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='x_output_ln')
    self.x_output_linear = tf.keras.layers.Dense(
        out_dim, name='x_output_linear')

  @property
  def hidden_shapes(self):
    return [[self.num_groups, self.latents_per_group, self.latent_dim]]

  def initialize_cond(self, t, cond, training):
    t = tf.expand_dims(self.time_emb(t, last_swish=False, normalize=True), 1)
    cond = self.cond_proj(cond)
    if cond.shape.ndims == 2:
      cond = tf.expand_dims(cond, 1)
    return t, cond

  def initialize_latent(self, batch_size, latents_prev, training):
    latents = tf.reshape(self.latent_pos_emb,
                         [1, self.num_groups, self.latents_per_group, -1])
    latents = tf.tile(latents, [batch_size, 1, 1, 1])
    if self.self_cond in ['latent']:
      latents += self.latent_prev_ln(self.latent_prev_proj(latents_prev))
    return latents

  def call(self, x, time, cond, training=True):
    """x[0] in (bsz, t, n, c), time in (bsz, m), cond in (bsz, s, d)."""
    if isinstance(x, tuple) or isinstance(x, list):
      x, latents_prev = x
      bsz = tf.shape(x)[0]
    else:
      bsz = tf.shape(x)[0]
      latents_prev = tf.zeros([bsz] + self.hidden_shapes[0])
    bsz = tf.shape(x)[0]
    t = self.num_groups
    x = self.stem_ln(self.stem(x))
    x = x + self.x_pos_emb[tf.newaxis, tf.newaxis, ...]
    latents = self.initialize_latent(bsz, latents_prev, training)
    time, cond = self.initialize_cond(time, cond, training)
    cond = tf.concat([time, cond], 1)

    latents = einops.rearrange(latents, 'b t m d -> (b t) m d', t=t)
    x = einops.rearrange(x, 'b t n c -> (b t) n c')
    for i in range(len(self.layer_configs)):
      x = self.x_network[str(i)](x, None, training=training)
      if self.layer_configs[i][-1] > 0:
        latents = self.l2x_cross_attn[str(i)](
            latents, x, None, None, None, training=training)[0]
        latents = einops.rearrange(latents, '(b t) m d -> b (t m) d', t=t)
        latents = self.l2c_cross_attn[str(i)](
            latents, cond, None, None, None, training=training)[0]
        latents = self.l_network[str(i)](latents, self.mask, training)
        latents = einops.rearrange(latents, 'b (t m) d -> (b t) m d', t=t)
        if i < len(self.layer_configs) - 1:
          x = self.x2l_cross_attn[str(i)](
              x, latents, None, None, None, training=training)[0]

    x = einops.rearrange(x, '(b t) m d -> b t m d', t=t)
    latents = einops.rearrange(latents, '(b t) m d -> b t m d', t=t)
    return self.x_output_linear(self.x_output_ln(x)), latents


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


class FITAR(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               layers,  # str format: (local-layers)global-layers.. eg '(2)4(2)'
               x_size,
               num_groups,
               latents_per_group,
               x_dim,
               latent_dim,
               x_num_heads,
               latent_num_heads,
               mlp_ratio,
               vocab_size,
               shared_embedding=True,
               output_bias=True,
               drop_path=0.0,
               drop_units=0.0,
               drop_att=0.0,
               x_pos_encoding='learned',
               latent_pos_encoding='learned',
               **kwargs):
    super().__init__(**kwargs)
    if x_size % num_groups != 0:
      raise ValueError(
          f'x_size={x_size} is not divisible by num_groups={num_groups}')
    x_per_group = x_size // num_groups
    self.num_groups = num_groups
    self.latents_per_group = latents_per_group
    self.shared_embedding = shared_embedding
    self.output_bias = output_bias
    self.layer_configs = get_layer_config(layers)
    self.x_mask = 1. - get_ar_mask(x_per_group, dtype=tf.float32)
    self.l_mask = 1. - get_chunk_ar_mask(
        num_groups * latents_per_group, latents_per_group, dtype=tf.float32)

    self.latent_pos_emb = add_vis_pos_emb(
        self, latent_pos_encoding, num_groups, latents_per_group, latent_dim,
        name_prefix=f'{self.name}/latent_pos_emb/kernel',
        return_only=True, normalization_max=1000.)
    self.x_pos_emb = add_vis_pos_emb(
        self, x_pos_encoding, x_per_group, 1, x_dim,
        name_prefix=f'{self.name}/x_pos_emb/kernel',
        return_only=True, normalization_max=1000.)
    add_vocab_token_emb(
        self, vocab_size, x_dim, shared_embedding, output_bias)

    self.x2l_cross_attn = {}
    self.l2x_cross_attn = {}
    self.x_network = {}
    self.l_network = {}
    for i, (x_layers, l_layers) in enumerate(self.layer_configs):
      self.x_network[str(i)] = TransformerDecoder(
          x_layers,
          dim=x_dim,
          mlp_ratio=mlp_ratio,
          num_heads=x_num_heads,
          drop_path=drop_path,
          drop_units=drop_units,
          drop_att=drop_att,
          cross_attention=False,
          name='x_network' + suffix_id(i))
      if l_layers > 0:
        self.l2x_cross_attn[str(i)] = TransformerDecoderLayer(
            dim=0,
            mlp_ratio=0,
            num_heads=min(latent_num_heads, x_num_heads),
            drop_path=0.,
            drop_units=0.,
            drop_att=0.,
            dim_x_att=min(latent_dim, x_dim),
            self_attention=False,
            cross_attention=True,
            use_mlp=False,
            use_enc_ln=True,
            name='l2x_cross_attn' + suffix_id(i))
        self.l_network[str(i)] = TransformerDecoder(
            l_layers,
            dim=latent_dim,
            mlp_ratio=mlp_ratio,
            num_heads=latent_num_heads,
            drop_path=drop_path,
            drop_units=drop_units,
            drop_att=drop_att,
            cross_attention=False,
            name='l_network' + suffix_id(i))
      if i < len(self.layer_configs) - 1:
        self.x2l_cross_attn[str(i)] = TransformerDecoderLayer(
            dim=0,
            mlp_ratio=0,
            num_heads=min(x_num_heads, latent_num_heads),
            drop_path=0.,
            drop_units=0.,
            drop_att=0.,
            dim_x_att=min(latent_dim, x_dim),
            self_attention=False,
            cross_attention=True,
            use_mlp=False,
            use_enc_ln=True,
            name='x2l_cross_attn' + suffix_id(i))
    self.x_output_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='x_output_ln')
    self.l_output_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='l_output_ln')

  def _latent_shift(self, latents, s_len):
    """latents shape change: b t m d -> (b t) m d."""
    latents_leading, latents_last = latents[:, :-1], latents[:, -1:]
    latents = tf.concat([tf.zeros_like(latents_last), latents_leading], axis=1)
    latents = einops.rearrange(latents, 'b t m d -> (b t) m d', t=s_len)
    return latents, latents_last

  def _latent_shift_back(self, latents, latents_last, s_len):
    """latents shape change: (b t) m d -> b t m d."""
    latents = einops.rearrange(latents, '(b t) m d -> b t m d', t=s_len)
    latents = tf.concat([latents[:, 1:], latents_last], axis=1)
    return latents

  def call(self, x, encoded=None, training=True):
    """x (e.g. token id) is an integer tensor with shape of [bsz, t, n]."""
    del encoded  # not implemented.
    bsz = tf.shape(x)[0]
    t = self.num_groups
    x_mask, l_mask = self.x_mask, self.l_mask
    if self.shared_embedding:
      inp_embedding = outp_embedding = self.token_embedding
    else:
      inp_embedding = self.inp_token_embedding
      outp_embedding = self.outp_token_embedding
    x = tf.gather(inp_embedding, x)
    x = x + self.x_pos_emb[tf.newaxis, tf.newaxis, ...]
    latents = tf.reshape(self.latent_pos_emb,
                         [1, t, self.latents_per_group, -1])
    latents = tf.tile(latents, [bsz, 1, 1, 1])

    x = einops.rearrange(x, 'b t n c -> (b t) n c')
    for i in range(len(self.layer_configs)):
      x = self.x_network[str(i)](
          x, None, None, x_mask, None, training=training)[0]

      if self.layer_configs[i][-1] > 0:
        latents = einops.rearrange(latents, 'b t m d -> (b t) m d')
        latents = self.l2x_cross_attn[str(i)](
            latents, x, None, None, None, training=training)[0]
        latents = einops.rearrange(latents, '(b t) m d -> b (t m) d', t=t)
        latents = self.l_network[str(i)](
            latents, None, None, l_mask, None, training=training)[0]
        latents = einops.rearrange(latents, 'b (t m) d -> b t m d', t=t)
        if i < len(self.layer_configs) - 1:
          latents, latents_last = self._latent_shift(latents, t)
          x = self.x2l_cross_attn[str(i)](
              x, latents, None, None, None, training=training)[0]
          latents = self._latent_shift_back(latents, latents_last, t)

    x = einops.rearrange(x, '(b t) n d -> b t n d', t=t)
    logits = tf.einsum('btnd,kd->btnk', self.x_output_ln(x), outp_embedding)
    if self.output_bias:
      logits = tf.nn.bias_add(logits, self.outp_bias)
    return logits
