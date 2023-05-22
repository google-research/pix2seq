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
"""Recurrent Interface Network (RIN), but here it is named Tape."""

import einops
from architectures.convnet_blocks import DepthwiseConvBlock
from architectures.transformers import add_vis_pos_emb
from architectures.transformers import get_shape
from architectures.transformers import MLP
from architectures.transformers import TransformerDecoder
from architectures.transformers import TransformerEncoder
from architectures.transunet import ScalarEmbedding
import tensorflow as tf


class TapeDenoiser(tf.keras.layers.Layer):  # pylint: disable=missing-docstring
  """This is an abstract class."""

  def __init__(self,
               num_layers,
               latent_slots,
               latent_dim,
               latent_mlp_ratio,
               latent_num_heads,
               tape_dim,
               tape_mlp_ratio,
               rw_num_heads,
               conv_kernel_size=0,
               conv_drop_units=0,
               latent_pos_encoding='learned',
               tape_pos_encoding='learned',
               drop_path=0.,
               drop_units=0.1,
               drop_att=0.,
               time_scaling=1e4,
               self_cond='none',
               time_on_latent=False,
               cond_on_latent_n=0,
               cond_tape_writable=False,
               cond_dim=0,
               cond_proj=True,
               cond_decoupled_read=False,
               xattn_enc_ln=False,
               **kwargs):
    super().__init__(**kwargs)
    self._num_layers = [int(i) for i in num_layers.split(',')]
    self._latent_slots = latent_slots
    self._time_on_latent = time_on_latent
    self._cond_on_latent = cond_on_latent_n > 0
    if self._time_on_latent:  # replace 1 latent with time emb.
      latent_slots -= 1
    latent_slots -= cond_on_latent_n
    self._latent_dim = latent_dim
    # TODO(iamtingchen): the slots are inaccurate when cond is not None.
    self._tape_slots = self._num_tokens
    self._tape_dim = tape_dim
    self._cond_dim = cond_dim = cond_dim if cond_dim > 0 else tape_dim
    self._latent_pos_encoding = latent_pos_encoding
    self._tape_pos_encoding = tape_pos_encoding
    self._self_cond = self_cond
    self._cond_tape_writable = cond_tape_writable
    self._cond_decoupled_read = cond_decoupled_read
    time_scaling = tf.constant(time_scaling, dtype=tf.float32)
    assert self_cond in ['none', 'latent', 'latent+tape', 'tape']
    self.stem_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='stem_ln')
    self.time_emb = ScalarEmbedding(
        dim=(latent_dim if self._time_on_latent else cond_dim) // 4,
        scaling=time_scaling,
        expansion=4,
        name='time_emb')
    if cond_proj:
      self.cond_proj = tf.keras.layers.Dense(
          latent_dim if self._cond_on_latent else cond_dim, name='cond_proj')
    else:
      self.cond_proj = lambda x: x

    self.make_latent_pos(latent_slots, latent_dim, latent_pos_encoding,
                         time_scaling)
    self.make_tape_pos(tape_dim, tape_pos_encoding, time_scaling)

    if self_cond in ['latent', 'latent+tape']:
      self.latent_prev_proj = MLP(
          num_layers=1,
          dim=latent_dim,
          mlp_ratio=latent_mlp_ratio,
          drop_path=0.,
          drop_units=0.,
          name='latent_prev_proj')
      self.latent_prev_ln = tf.keras.layers.LayerNormalization(
          epsilon=1e-6, gamma_initializer='zeros', name='latent_prev_ln')
    if self_cond in ['tape', 'latent+tape']:
      self.tape_prev_proj = MLP(
          num_layers=1,
          dim=tape_dim,
          mlp_ratio=tape_mlp_ratio,
          drop_path=0.,
          drop_units=0.,
          name='tape_prev_proj')
      self.tape_prev_ln = tf.keras.layers.LayerNormalization(
          epsilon=1e-6, gamma_initializer='zeros', name='tape_prev_ln')
    self.read_units = {}
    self.read_cond_units = {}
    self.write_units = {}
    self.conv_units = {}
    self.latent_processing_units = {}
    for i, num_layers_per_readwrite in enumerate(self._num_layers):
      self.read_units[str(i)] = TransformerDecoder(
          num_layers=1,
          dim=latent_dim,
          mlp_ratio=latent_mlp_ratio,
          num_heads=rw_num_heads,
          drop_path=0.,
          drop_units=0.,
          drop_att=0.,
          dim_x_att=min(tape_dim, latent_dim),
          self_attention=False,
          cross_attention=True,
          use_mlp=True,
          use_enc_ln=xattn_enc_ln,
          name=f'read_unit_{i}')
      if cond_decoupled_read:
        self.read_cond_units[str(i)] = TransformerDecoder(
            num_layers=1,
            dim=latent_dim,
            mlp_ratio=latent_mlp_ratio,
            num_heads=rw_num_heads,
            drop_path=0.,
            drop_units=0.,
            drop_att=0.,
            dim_x_att=min(cond_dim, latent_dim),
            self_attention=False,
            cross_attention=True,
            use_mlp=True,
            use_enc_ln=xattn_enc_ln,
            name=f'read_cond_unit_{i}')
      if num_layers_per_readwrite == 0:
        self.write_units[str(i)] = lambda x, *args, **kwargs: (x, None)
        self.conv_units[str(i)] = lambda x, *args, **kwargs: x
        self.latent_processing_units[str(i)] = lambda x, *args, **kwargs: x
      else:
        self.write_units[str(i)] = TransformerDecoder(
            num_layers=1,
            dim=tape_dim,
            mlp_ratio=tape_mlp_ratio,
            num_heads=rw_num_heads,
            drop_path=0.,
            drop_units=0.,
            drop_att=0.,
            dim_x_att=min(tape_dim, latent_dim),
            self_attention=False,
            cross_attention=True,
            use_mlp=True if tape_mlp_ratio > 0 else False,
            use_enc_ln=xattn_enc_ln,
            name=f'write_unit_{i}')
        if conv_kernel_size == 0:
          self.conv_units[str(i)] = lambda x, *args, **kwargs: x
        else:
          self.conv_units[str(i)] = DepthwiseConvBlock(
              tape_dim,
              kernel_size=conv_kernel_size,
              dropout_rate=conv_drop_units,
              name=f'conv_units_{i}')
        self.latent_processing_units[str(i)] = TransformerEncoder(
            num_layers=num_layers_per_readwrite,
            dim=latent_dim,
            mlp_ratio=latent_mlp_ratio,
            num_heads=latent_num_heads,
            drop_path=drop_path,
            drop_units=drop_units,
            drop_att=drop_att,
            name=f'latent_processing_unit{i}')
    self.output_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
        center=True,
        scale=True,
        name='output_ln')
    self.output_linear = tf.keras.layers.Dense(
        self._output_dim, name='output_linear')

  def make_latent_pos(self,
                      latent_slots,
                      latent_dim,
                      latent_pos_encoding,
                      time_scaling):
    if latent_pos_encoding in ['sin_cos_plus_learned']:
      self.latent_pos_emb = add_vis_pos_emb(
          self, 'sin_cos', latent_slots, 1, latent_dim,
          name_prefix=f'{self.name}/latent_pos_emb/kernel',
          return_only=True, normalization_max=time_scaling)
      self.latent_pos_emb_res = self.add_weight(
          shape=(latent_slots, latent_dim), initializer='zeros',
          name=f'{self.name}/latent_pos_emb_res/kernel')
    elif latent_pos_encoding in ['learned', 'sin_cos']:
      self.latent_pos_emb = add_vis_pos_emb(
          self, latent_pos_encoding, latent_slots, 1, latent_dim,
          name_prefix=f'{self.name}/latent_pos_emb/kernel',
          return_only=True, normalization_max=time_scaling)
    else:
      raise ValueError(f'Unknown latent_pos_encoding {latent_pos_encoding}')

  def make_tape_pos(self, tape_dim, tape_pos_encoding, time_scaling):
    if tape_pos_encoding in ['sin_cos_plus_learned']:
      self.tape_pos_emb = add_vis_pos_emb(
          self, 'sin_cos', self._n_rows, self._n_cols, tape_dim,
          name_prefix=f'{self.name}/tape_pos_emb/kernel',
          return_only=True, normalization_max=time_scaling)
      self.tape_pos_emb_res = self.add_weight(
          shape=(self._n_rows * self._n_cols, tape_dim), initializer='zeros',
          name=f'{self.name}/tape_pos_emb_res/kernel')
    elif tape_pos_encoding in ['learned', 'sin_cos']:
      self.tape_pos_emb = add_vis_pos_emb(
          self, tape_pos_encoding, self._n_rows, self._n_cols, tape_dim,
          name_prefix=f'{self.name}/tape_pos_emb/kernel', return_only=True,
          normalization_max=time_scaling)
    else:
      raise ValueError(f'Unknown tape_pos_encoding {tape_pos_encoding}')

  def initialize_cond(self, t, cond, training):
    if t is not None:
      t = tf.expand_dims(self.time_emb(t, last_swish=False, normalize=True), 1)
    if cond is not None:
      cond = self.cond_proj(cond)
      if cond.shape.ndims == 2:
        cond = tf.expand_dims(cond, 1)
    return t, cond

  def initialize_tape(self, x, time_emb, cond, tape_prev, offset=0):
    tape_r = None
    if not self._time_on_latent and time_emb is not None:
      tape_r = time_emb
    if not self._cond_on_latent and cond is not None:
      tape_r = cond if tape_r is None else tf.concat([tape_r, cond], 1)
    tape = self._x_to_tape(x, offset)  # (bsz, n, d)

    if self._self_cond in ['tape', 'latent+tape'] and tape_prev is not None:
      tape += self.tape_prev_ln(self.tape_prev_proj(tape_prev))
    if self._cond_tape_writable and tape_r is not None:
      tape, tape_r = tf.concat([tape, tape_r], 1), None
    return tape, tape_r

  def initialize_latent(self, batch_size, time_emb, cond, latent_prev):
    latent = self.latent_pos_emb[tf.newaxis, ...]
    if self._latent_pos_encoding in ['sin_cos_plus_learned']:
      latent += self.latent_pos_emb_res[tf.newaxis, ...]
    latent = tf.tile(latent, [batch_size, 1, 1])
    if self._time_on_latent and time_emb is not None:
      latent = tf.concat([latent, time_emb], 1)
    if self._cond_on_latent and cond is not None:
      latent = tf.concat([latent, cond], 1)
    if self._self_cond in ['latent', 'latent+tape']:
      latent += self.latent_prev_ln(self.latent_prev_proj(latent_prev))
    return latent

  def _merge_tape(self, tape_writable, tape_readonly):
    tape_merged = tape_writable if tape_readonly is None else (
        tf.concat([tape_writable, tape_readonly], 1))
    return tape_merged

  def compute(self, latent, tape, tape_r, training):
    for i in range(len(self._num_layers)):
      if self._cond_decoupled_read:
        latent = self.read_cond_units[str(i)](
            latent, tape_r, None, None, None, training)[0]
        latent = self.read_units[str(i)](
            latent, tape, None, None, None, training)[0]
      else:
        tape_merged = self._merge_tape(tape, tape_r)
        latent = self.read_units[str(i)](
            latent, tape_merged, None, None, None, training)[0]
      latent = self.latent_processing_units[str(i)](latent, None, training)
      tape = self.write_units[str(i)](
          tape, latent, None, None, None, training)[0]
      tape = self.conv_units[str(i)](
          tape, training, size=self._num_tokens)
    return latent, tape

  def readout_tape(self, tape):
    tokens = self.output_linear(self.output_ln(tape[:, :self._num_tokens]))
    return tokens

  @property
  def hidden_shapes(self):
    latent_shape = [self._latent_slots, self._latent_dim]
    tape_shape = [self._tape_slots, self._tape_dim]
    return latent_shape, tape_shape

  def call(self, x, t, cond, training):
    """x[0] in (bsz, h, w, c), t in (bsz, m), cond in (bsz, s, d)."""
    if isinstance(x, tuple) or isinstance(x, list):
      x, latent_prev, tape_prev = x
      bsz = tf.shape(x)[0]
    else:
      bsz = tf.shape(x)[0]
      latent_prev = tf.zeros([bsz] + self.hidden_shapes[0])
      tape_prev = tf.zeros([bsz] + self.hidden_shapes[1])
    time_emb, cond = self.initialize_cond(t, cond, training)
    tape, tape_r = self.initialize_tape(x, time_emb, cond, tape_prev)
    latent = self.initialize_latent(bsz, time_emb, cond, latent_prev)
    latent, tape = self.compute(latent, tape, tape_r, training)
    x = self.readout_tape(tape)
    return x, latent, tape[:, :self._tape_slots]


class TokenTapeDenoiser(TapeDenoiser):  # pylint: disable=missing-docstring
  """Deal with token data of shape (bsz, seqlen, d)."""

  def __init__(self,
               num_layers,
               latent_slots,
               latent_dim,
               latent_mlp_ratio,
               latent_num_heads,
               tape_dim,
               tape_mlp_ratio,
               rw_num_heads,
               num_tokens,
               token_output_dim,
               latent_pos_encoding='learned',
               tape_pos_encoding='learned',
               drop_path=0.,
               drop_units=0.1,
               drop_att=0.,
               time_scaling=1e4,
               self_cond='none',
               **kwargs):
    self._num_tokens = num_tokens
    if tape_pos_encoding in ['sin_cos', 'sin_cos_plus_learned']:
      self._n_rows = self._n_cols = tf.cast(
          tf.math.sqrt(tf.cast(num_tokens, tf.float32)), tf.int32)
    else:
      self._n_rows, self._n_cols = num_tokens, 1
    self._output_dim = token_output_dim
    super().__init__(
        num_layers=num_layers,
        latent_slots=latent_slots,
        latent_dim=latent_dim,
        latent_mlp_ratio=latent_mlp_ratio,
        latent_num_heads=latent_num_heads,
        tape_dim=tape_dim,
        tape_mlp_ratio=tape_mlp_ratio,
        rw_num_heads=rw_num_heads,
        latent_pos_encoding=latent_pos_encoding,
        tape_pos_encoding=tape_pos_encoding,
        drop_path=drop_path,
        drop_units=drop_units,
        drop_att=drop_att,
        time_scaling=time_scaling,
        self_cond=self_cond,
        **kwargs)

    self.stem = tf.keras.layers.Dense(tape_dim, name='stem')

  def _x_to_tape(self, x, offset=0):
    tokens = self.stem(x)
    tape_pos_emb = self.tape_pos_emb[tf.newaxis, ...]
    if self._tape_pos_encoding in ['sin_cos_plus_learned']:
      tape_pos_emb += self.tape_pos_emb_res[tf.newaxis, ...]
    tokens = self.stem_ln(tokens) + tape_pos_emb
    return tokens


class ImageTapeDenoiser(TapeDenoiser):  # pylint: disable=missing-docstring
  """Deal with image data of shape (bsz, h, w, c)."""

  def __init__(self,
               num_layers,
               latent_slots,
               latent_dim,
               latent_mlp_ratio,
               latent_num_heads,
               tape_dim,
               tape_mlp_ratio,
               rw_num_heads,
               image_height,
               image_width,
               image_channels,
               patch_size,
               latent_pos_encoding='learned',
               tape_pos_encoding='learned',
               drop_path=0.,
               drop_units=0.1,
               drop_att=0.,
               time_scaling=1e4,
               self_cond='none',
               **kwargs):
    self._n_rows = image_height // patch_size
    self._n_cols = image_width // patch_size
    self._num_tokens = self._n_rows * self._n_cols
    self._patch_size = patch_size
    self._output_dim = patch_size**2 * image_channels
    super().__init__(
        num_layers=num_layers,
        latent_slots=latent_slots,
        latent_dim=latent_dim,
        latent_mlp_ratio=latent_mlp_ratio,
        latent_num_heads=latent_num_heads,
        tape_dim=tape_dim,
        tape_mlp_ratio=tape_mlp_ratio,
        rw_num_heads=rw_num_heads,
        latent_pos_encoding=latent_pos_encoding,
        tape_pos_encoding=tape_pos_encoding,
        drop_path=drop_path,
        drop_units=drop_units,
        drop_att=drop_att,
        time_scaling=time_scaling,
        self_cond=self_cond,
        **kwargs)

    self.stem = tf.keras.layers.Conv2D(
        filters=tape_dim, kernel_size=patch_size, strides=patch_size,
        padding='VALID', use_bias=True, name='stem')

  def _x_to_tape(self, x, offset=0):
    tokens = self.stem(x)
    bsz, h, w, d = get_shape(tokens)
    tokens = tf.reshape(tokens, [bsz, h * w, d])
    tape_pos_emb = self.tape_pos_emb[tf.newaxis, ...]
    if self._tape_pos_encoding in ['sin_cos_plus_learned']:
      tape_pos_emb += self.tape_pos_emb_res[tf.newaxis, ...]
    tokens = self.stem_ln(tokens) + tape_pos_emb
    return tokens

  def readout_tape(self, tape):
    tokens = super().readout_tape(tape)
    bsz, _, d = get_shape(tokens)
    tokens = tf.reshape(tokens, [bsz, self._n_rows, self._n_cols, d])
    if self._patch_size == 1:
      return tokens
    else:
      return tf.nn.depth_to_space(tokens, self._patch_size)


class VideoTapeDenoiser(TapeDenoiser):  # pylint: disable=missing-docstring
  """Deal with video data of shape (bsz, t, h, w, c)."""

  def __init__(self,
               num_layers,
               latent_slots,
               latent_dim,
               latent_mlp_ratio,
               latent_num_heads,
               tape_dim,
               tape_mlp_ratio,
               rw_num_heads,
               image_height,
               image_width,
               image_channels,
               patch_size,
               seq_len,
               seq_stride,
               seq_cond=0,
               latent_pos_encoding='learned',
               tape_pos_encoding='learned',
               drop_path=0.,
               drop_units=0.1,
               drop_att=0.,
               time_scaling=1e4,
               self_cond='none',
               **kwargs):
    self._n_rows = image_height // patch_size
    self._n_cols = image_width // patch_size
    t_pad = (seq_len-seq_cond) % seq_stride + seq_cond % seq_stride
    self._n_time = (seq_len + t_pad) // seq_stride     # odd-len pad
    self._num_tokens = self._n_time * self._n_rows * self._n_cols
    self._patch_size = patch_size
    self._seq_len = seq_len
    self._seq_stride = seq_stride
    self._seq_cond = seq_cond
    self._output_dim = seq_stride * patch_size**2 * image_channels
    super().__init__(
        num_layers=num_layers,
        latent_slots=latent_slots,
        latent_dim=latent_dim,
        latent_mlp_ratio=latent_mlp_ratio,
        latent_num_heads=latent_num_heads,
        tape_dim=tape_dim,
        tape_mlp_ratio=tape_mlp_ratio,
        rw_num_heads=rw_num_heads,
        latent_pos_encoding=latent_pos_encoding,
        tape_pos_encoding=tape_pos_encoding,
        drop_path=drop_path,
        drop_units=drop_units,
        drop_att=drop_att,
        time_scaling=time_scaling,
        self_cond=self_cond,
        **kwargs)

    self.stem = tf.keras.layers.Conv3D(
        filters=tape_dim,
        kernel_size=(seq_stride, patch_size, patch_size),
        strides=(seq_stride, patch_size, patch_size),
        padding='SAME',
        use_bias=True,
        name='stem'
    )

  def make_tape_pos(self, tape_dim, tape_pos_encoding, time_scaling):
    if tape_pos_encoding in ['sin_cos_plus_learned']:
      self.tape_pos_emb = add_vis_pos_emb(
          self, 'sin_cos', self._n_rows, self._n_cols, tape_dim,
          name_prefix=f'{self.name}/tape_pos_emb/kernel',
          return_only=True, normalization_max=time_scaling)
      self.tape_pos_emb = einops.repeat(self.tape_pos_emb,
                                        'hw d -> (t hw) d', t=self._n_time)
      self.tape_pos_emb_res = self.add_weight(
          shape=(self._n_time * self._n_rows * self._n_cols, tape_dim),
          initializer='zeros',
          name=f'{self.name}/tape_pos_emb_res/kernel')
    elif tape_pos_encoding == 'sin_cos':
      self.tape_pos_emb = add_vis_pos_emb(
          self, tape_pos_encoding, self._n_rows, self._n_cols, tape_dim,
          name_prefix=f'{self.name}/tape_pos_emb/kernel', return_only=True,
          normalization_max=time_scaling)
      self.tape_pos_emb = tf.repeat(self.tape_pos_emb, [self._n_time], 0)
    elif tape_pos_encoding == 'learned':
      self.tape_pos_emb = add_vis_pos_emb(
          self, tape_pos_encoding, self._n_time * self._n_rows, self._n_cols,
          tape_dim, name_prefix=f'{self.name}/tape_pos_emb/kernel',
          return_only=True, normalization_max=time_scaling)
    else:
      raise ValueError(f'Unknown tape_pos_encoding {tape_pos_encoding}')

  def _x_to_tape(self, x, offset=0):
    tokens = self.stem(x)
    bsz, t, h, w, d = get_shape(tokens)
    tokens = tf.reshape(tokens, [bsz, t * h * w, d])
    pos_emb = self.tape_pos_emb[tf.newaxis, ...]
    if self._tape_pos_encoding in ['sin_cos_plus_learned']:
      pos_emb += self.tape_pos_emb_res[tf.newaxis, ...]
    tokens = self.stem_ln(tokens) + pos_emb[:, offset:offset+tokens.shape[1]]
    return tokens

  def readout_tape(self, tape):
    tokens = super().readout_tape(tape)
    out = einops.rearrange(
        tokens,
        'b (t h w) (t1 h1 w1 c) -> b (t t1) (h h1) (w w1) c',
        h1=self._patch_size, w1=self._patch_size, t1=self._seq_stride,
        h=self._n_rows, w=self._n_cols
    )
    out_len = self._seq_len - self._seq_cond
    if out.shape[1] > out_len: out = out[:, :out_len]   # odd-len conv padding
    return out

  def call(self, x, t, cond, training):
    """x[0] in (bsz, t, h, w, c), t in (bsz, m), cond in (bsz, t_c, h, w, c)."""
    if isinstance(x, tuple) or isinstance(x, list):
      x, latent_prev, tape_prev = x
      bsz = tf.shape(x)[0]
    else:
      bsz = tf.shape(x)[0]
      latent_prev = tf.zeros([bsz] + self.hidden_shapes[0])
      tape_prev = tf.zeros([bsz] + self.hidden_shapes[1])
    time_emb, _ = self.initialize_cond(t, None, training)
    tape, tape_r = self.initialize_tape(x, time_emb, None, tape_prev)
    cond_offset = tape.shape[1]
    tape_cond, _ = self.initialize_tape(cond, None, None, None,
                                        offset=cond_offset)
    tape = tf.concat([tape, tape_cond], axis=1)
    latent = self.initialize_latent(bsz, time_emb, cond, latent_prev)
    latent, tape = self.compute(latent, tape, tape_r, training)
    # tape = tape[:, :cond_offset]
    x = self.readout_tape(tape)
    return x, latent, tape
