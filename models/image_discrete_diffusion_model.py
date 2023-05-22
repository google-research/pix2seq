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
"""The discrete/categorical image diffusion model."""

import ml_collections

import utils
from architectures.tape import ImageTapeDenoiser
from architectures.transunet import TransUNet
from models import diffusion_utils
from models import model as model_lib
import tensorflow as tf


@model_lib.ModelRegistry.register('image_discrete_diffusion_model')
class Model(tf.keras.models.Model):
  """A model."""

  def __init__(self, config: ml_collections.ConfigDict, **kwargs):
    super().__init__(**kwargs)
    image_size = config.dataset.image_size
    self.image_size = image_size
    self.num_classes = config.dataset.num_classes
    config = config.model
    self.config = config
    self.scheduler = diffusion_utils.Scheduler(config.train_schedule)
    if config.x0_clip == 'auto':
      self.x0_clip = '{},{}'.format(-config.b_scale, config.b_scale)
    else:
      self.x0_clip = config.x0_clip
    self.x_channels = get_x_channels(config.b_type)
    if config.arch_name == 'transunet':
      if ',' in config.kernel_sizes:
        kernel_sizes = [int(x) for x in config.kernel_sizes.split(',')]
      else:
        kernel_sizes = int(config.kernel_sizes)
      m_kwargs = dict(
          out_dim=self.x_channels,
          dim=config.dim,
          in_strides=config.in_strides,
          in_kernel_size=config.in_kernel_size,
          out_kernel_size=config.out_kernel_size,
          kernel_sizes=kernel_sizes,
          n_res_blocks=[int(x) for x in config.n_res_blocks.split(',')],
          ch_multipliers=[int(x) for x in config.ch_multipliers.split(',')],
          n_mlp_blocks=config.n_mlp_blocks,
          dropout=config.udrop,
          mhsa_resolutions=[int(x) for x in config.mhsa_resolutions.split(',')],
          per_head_dim=config.per_head_dim,
          transformer_dim=config.transformer_dim,
          transformer_strides=config.transformer_strides,
          transformer_blocks=config.transformer_blocks,
          conditioning=False,
          time_scaling=config.time_scaling,
          pos_encoding=config.u_pos_encoding,
          norm_type='group_norm',
          outp_softmax_groups=config.outp_softmax_groups,
          b_scale=config.b_scale,
          name=config.arch_name)
      self.denoiser = TransUNet(**m_kwargs)
      self.denoiser_ema = TransUNet(**m_kwargs, trainable=False)
    elif config.arch_name == 'tape':
      m_kwargs = dict(
          num_layers=config.num_layers,
          latent_slots=config.latent_slots,
          latent_dim=config.latent_dim,
          latent_mlp_ratio=config.latent_mlp_ratio,
          latent_num_heads=config.latent_num_heads,
          tape_dim=config.tape_dim,
          tape_mlp_ratio=config.tape_mlp_ratio,
          rw_num_heads=config.rw_num_heads,
          conv_kernel_size=config.conv_kernel_size,
          conv_drop_units=config.conv_drop_units,
          image_height=image_size,
          image_width=image_size,
          image_channels=self.x_channels,
          patch_size=config.patch_size,
          latent_pos_encoding=config.latent_pos_encoding,
          tape_pos_encoding=config.tape_pos_encoding,
          drop_path=config.drop_path,
          drop_units=config.drop_units,
          drop_att=config.drop_att,
          time_scaling=config.time_scaling,
          self_cond=config.self_cond,
          time_on_latent=config.time_on_latent,
          cond_on_latent_n=1 if config.cond_on_latent else 0,
          cond_tape_writable=config.cond_tape_writable,
          cond_dim=config.cond_dim,
          cond_proj=config.cond_proj,
          cond_decoupled_read=config.cond_decoupled_read,
          xattn_enc_ln=config.xattn_enc_ln,
          name=config.arch_name)
      self.denoiser = ImageTapeDenoiser(**m_kwargs)
      self.denoiser_ema = ImageTapeDenoiser(**m_kwargs, trainable=False)
    else:
      raise ValueError(f'Unknown architecture {config.arch_name}')
    # Obtain hidden shapes for latent self conditioning.
    # TODO(iamtingchen): better way to handle it.
    self.hidden_shapes = getattr(self.denoiser, 'hidden_shapes', None)
    if self.hidden_shapes is not None:  # latent self-cond
      assert config.self_cond not in ['x', 'eps', 'auto']

  def get_cond_denoise(self, labels, for_loss=False):
    config = self.config
    def cond_denoise(x, gamma, training, drop_label=False):
      gamma = tf.reshape(gamma, [-1])
      cond = None
      if config.conditional == 'class':
        cond_dropout = config.get('cond_dropout', 0.)
        labels_w = 1.
        if training and cond_dropout > 0:
          labels_w = tf.random.uniform([tf.shape(labels)[0], 1]) > cond_dropout
          labels_w = tf.cast(labels_w, tf.float32)
        if drop_label:
          labels_w = 0.
        if config.arch_name == 'transunet':  # Merge one-hot label with gamma.
          gamma = tf.concat([gamma[..., tf.newaxis], labels * labels_w], -1)
        else:
          cond = labels * labels_w
      elif config.conditional != 'none':
        raise ValueError(f'Unknown conditional {config.conditional}')
      return self.denoise(x, gamma, cond, training, for_loss=for_loss)
    return cond_denoise

  def denoise(self, x, gamma, cond, training, for_loss=False):
    """gamma should be (bsz, ) or (bsz, d)."""
    assert gamma.shape.rank <= 2
    config = self.config
    if not hasattr(self, 'denoise_x_shape'):
      if isinstance(x, tuple) or isinstance(x, list):
        self.denoise_x_shape = tuple(tf.shape(x_) for x_ in x)
      else:
        self.denoise_x_shape = tf.shape(x)
      self.denoise_gamma_shape = tf.shape(gamma)
      self.cond_shape = None if cond is None else tf.shape(cond)
    denoiser = self.denoiser if training else self.denoiser_ema
    if self.config.normalize_noisy_input:
      if isinstance(x, tuple) or isinstance(x, list):
        x = list(x)
        x[0] /= tf.math.reduce_std(
            x[0], list(range(1, x[0].shape.ndims)), keepdims=True)
      else:
        x /= tf.math.reduce_std(x, list(range(1, x.shape.ndims)), keepdims=True)
    x = denoiser(x, gamma, cond, training=training)
    if config.pred_type == 'x_sigmoid_xent':
      x = x if for_loss else (tf.nn.sigmoid(x) * 2 - 1) * config.b_scale
    if config.pred_type == 'x_softmax_xent':
      if isinstance(x, tuple) or isinstance(x, list):
        x = list(x)
        x[0] = unfold_rgb(x[0])
        x[0] = x[0] if for_loss else (
            fold_rgb(tf.nn.softmax(x[0]) * 2 - 1)) * config.b_scale
        x = tuple(x)
      else:
        x = unfold_rgb(x)
        x = x if for_loss else (
            fold_rgb(tf.nn.softmax(x)) * 2 - 1) * config.b_scale
    return x

  def sample(self, num_samples=100, iterations=100, method='ddim', **kwargs):
    config = self.config
    samples_shape = [
        num_samples, self.image_size, self.image_size, self.x_channels]
    if config.conditional == 'class':
      labels = tf.random.uniform(
          [num_samples], 0, self.num_classes, dtype=tf.int32)
      labels = tf.one_hot(labels, self.num_classes)
    else:
      labels = None
    samples = self.scheduler.generate(
        self.get_cond_denoise(labels),
        iterations,
        samples_shape,
        hidden_shapes=self.hidden_shapes,
        pred_type=config.pred_type,
        schedule=config.infer_schedule,
        td=config.td,
        x0_clip=self.x0_clip,
        self_cond=config.self_cond,
        guidance=config.guidance,
        sampler_name=method)
    return bit2rgb(samples, config.b_type)

  def noise_denoise(self, images, labels, time_step=None, training=True):
    config = self.config
    images = rgb2bit(images, config.b_type, config.b_scale, self.x_channels)
    images_noised, noise, _, gamma = self.scheduler.add_noise(
        images, time_step=time_step)
    if config.self_cond != 'none':
      sc_rate = config.get('self_cond_rate', 0.5)
      self_cond_by_masking = config.get('self_cond_by_masking', False)
      if self_cond_by_masking:
        sc_drop_rate = 1. - sc_rate
        num_sc_examples = tf.shape(images)[0]
      else:
        sc_drop_rate = 0.
        num_sc_examples = tf.cast(
            tf.cast(tf.shape(images)[0], tf.float32) * sc_rate, tf.int32)
      cond_denoise = self.get_cond_denoise(labels[:num_sc_examples])
      if self.hidden_shapes is None:  # data self-cond, return is a tensor.
        denoise_inputs = diffusion_utils.add_self_cond_estimate(
            images_noised, gamma, cond_denoise, config.pred_type,
            config.self_cond, self.x0_clip, num_sc_examples,
            drop_rate=sc_drop_rate, training=training)
      else:  # latent self-cond, return is a tuple.
        denoise_inputs = diffusion_utils.add_self_cond_hidden(
            images_noised, gamma, cond_denoise, num_sc_examples,
            self.hidden_shapes, drop_rate=sc_drop_rate, training=training)
    else:
      denoise_inputs = images_noised
    cond_denoise = self.get_cond_denoise(labels, for_loss=True)
    denoise_out = cond_denoise(denoise_inputs, gamma, training=training)
    if isinstance(denoise_out, tuple): denoise_out = denoise_out[0]
    return images, noise, images_noised, denoise_out

  def compute_loss(self,
                   images: tf.Tensor,
                   noise: tf.Tensor,
                   denoise_out: tf.Tensor) -> tf.Tensor:
    config = self.config
    if config.pred_type == 'x':
      loss = tf.reduce_mean(tf.square(images - denoise_out))
    elif config.pred_type == 'x_sigmoid_xent':
      pp = tf.nn.sigmoid(denoise_out * images / config.b_scale)
      loss = tf.reduce_mean(-tf.math.log(pp + 1e-8))
    elif config.pred_type == 'x_softmax_xent':
      images = (images / config.b_scale + 1) / 2.  # [-b, b] --> [0, 1]
      images = unfold_rgb(images)
      losses = tf.nn.softmax_cross_entropy_with_logits(images, denoise_out)
      loss = tf.reduce_mean(losses)
    elif config.pred_type == 'eps':
      loss = tf.reduce_mean(tf.square(noise - denoise_out))
    else:
      raise ValueError(f'Unknown pred_type {config.pred_type}')
    return loss

  def call(self,
           images: tf.Tensor,
           labels: tf.Tensor,
           training: bool = True,
           **kwargs) -> tf.Tensor:  # pylint: disable=signature-mismatch
    """Model inference call."""
    with tf.name_scope(''):  # for other functions to have the same name scope
      images, noise, _, denoise_out = self.noise_denoise(
          images, labels, None, training)
      return self.compute_loss(images, noise, denoise_out)


@model_lib.TrainerRegistry.register('image_discrete_diffusion_model')
class Trainer(model_lib.Trainer):
  """A trainer."""

  def compute_loss(self, preprocess_outputs):
    """Compute loss based on model outputs and targets."""
    images, labels = preprocess_outputs
    loss = self.model(images, labels, training=True)
    return loss

  def train_step(self, examples, tasks, strategy):
    super().train_step(examples, tasks, strategy)

    # EMA udpate
    oconfig = self.config.optimization
    ema_decay = oconfig.get('ema_decay', 0.)
    vars_src = self.model.denoiser.variables
    if not hasattr(self.model, 'ema_initialized'):
      self.model.ema_initialized = True
      if isinstance(self.model.denoise_x_shape, tuple):
        x = tuple(tf.zeros(sh) for sh in self.model.denoise_x_shape)
      else:
        x = tf.zeros(self.model.denoise_x_shape)
      if self.model.cond_shape is None:
        cond = None
      else:
        cond = tf.zeros(self.model.cond_shape)
      _ = self.model.denoise(
          x=x,
          gamma=tf.zeros(self.model.denoise_gamma_shape),
          cond=cond,
          training=False)
    vars_dst = self.model.denoiser_ema.variables
    assert len(vars_src) == len(vars_dst), (len(vars_src), len(vars_dst))
    if oconfig.get('ema_name_exact_match', False):
      src_vars_dict = dict((var.name, var) for var in vars_src)
      vars_src = [src_vars_dict[var.name] for var in vars_dst]
    for var_src, var_dst in zip(vars_src, vars_dst):
      var_dst.assign(var_dst * ema_decay + var_src * (1. - ema_decay))


class GrayCode:
  """Gray code converter."""

  def __init__(self, n_bits):
    self._n = n_bits
    self.codes = tf.constant(self.gen_gray_codes(n_bits), dtype=tf.int32)
    self.inv_codes = tf.math.invert_permutation(self.codes)

  def to_gray_code_tensor(self, x: tf.Tensor) -> tf.Tensor:
    return tf.gather(self.codes, tf.cast(x, tf.int32))

  def from_gray_code_tensor(self, x: tf.Tensor) -> tf.Tensor:
    return tf.gather(self.inv_codes, tf.cast(x, tf.int32))

  def gen_gray_codes(self, n):
    assert n > 0
    if n == 1:
      return [0, 1]
    shorter_gray_codes = self.gen_gray_codes(n - 1)
    bitmask = 1 << (n - 1)
    gray_codes = list(shorter_gray_codes)
    for gray_code in reversed(shorter_gray_codes):
      gray_codes.append(bitmask | gray_code)
    return gray_codes


# pylint: disable=bad-whitespace,bad-continuation,missing-function-docstring
def get_perm_inv_perm(b_type: str):
  if b_type == 'uint8_s':
    # perm/inv_perm generation
    # np.random.seed(42)
    # perm = np.arange(256)
    # np.random.shuffle(perm)
    # perm = tf.constant(perm, dtype=tf.int32)
    # inv_perm = tf.math.invert_permutation(perm)
    perm = tf.constant(
      [228,   6,  79, 206, 117, 185, 242, 167,   9,  30, 180, 222, 230,
       217, 136,  68, 199,  15,  96,  24, 235,  19, 120, 152,  33, 124,
       253, 208,  10, 164, 184,  97, 148, 190, 223,  25,  86,  18,  75,
       137, 196, 176, 239, 181,  45,  66,  16,  67, 215, 201, 177,  38,
       143,  84,  55, 220, 104, 139, 127,  60, 101, 172, 245, 126, 225,
       144, 108, 178,  73, 114, 158,  69, 141, 109, 115, 246, 113, 243,
        90,  29, 170,  82, 111,   5,  56, 132, 154, 162,  65, 186,  85,
       219, 237,  31,  12,  35,  28,  42, 112,  22, 125,  93, 173, 251,
        51, 240,  95, 146, 204,  76,  41, 119, 155,  78, 150,  26, 247,
       168, 118, 193, 140,   0,   2,  77,  46, 100, 205, 159, 183, 254,
        98,  36,  61, 200, 142,  11, 250, 224,  27, 231,   4, 122,  32,
       147, 182, 138,  62, 135, 128, 232, 194,  70, 197,  64,  44, 165,
       156,  40, 123, 153,  23, 192, 249,  81,  39, 244,  47,  94, 195,
       161,  43, 145, 175,   3, 105,  53, 133, 233, 198, 238,  49, 163,
        80,  34, 211,   7, 171, 216, 110,  91,  83, 229, 234,  89,   8,
        13,  59, 221, 131,  17, 166,  72, 226, 134, 209, 236,  63,  54,
       107,  50, 212, 174, 213, 189, 252, 207, 227, 169,  58, 218,  48,
        88,  21,  57, 203, 160, 248, 187, 191, 129,  37, 157, 241,   1,
        52, 149, 130, 151, 103,  99, 116,  87, 202,  74, 214, 210, 121,
       255,  20, 188,  71, 106,  14,  92, 179, 102], dtype=tf.int32)

    inv_perm = tf.constant(
      [121, 233, 122, 173, 140,  83,   1, 185, 194,   8,  28, 135,  94,
       195, 252,  17,  46, 199,  37,  21, 248, 222,  99, 160,  19,  35,
       115, 138,  96,  79,   9,  93, 142,  24, 183,  95, 131, 230,  51,
       164, 157, 110,  97, 170, 154,  44, 124, 166, 220, 180, 209, 104,
       234, 175, 207,  54,  84, 223, 218, 196,  59, 132, 146, 206, 153,
        88,  45,  47,  15,  71, 151, 250, 201,  68, 243,  38, 109, 123,
       113,   2, 182, 163,  81, 190,  53,  90,  36, 241, 221, 193,  78,
       189, 253, 101, 167, 106,  18,  31, 130, 239, 125,  60, 255, 238,
        56, 174, 251, 208,  66,  73, 188,  82,  98,  76,  69,  74, 240,
         4, 118, 111,  22, 246, 141, 158,  25, 100,  63,  58, 148, 229,
       236, 198,  85, 176, 203, 147,  14,  39, 145,  57, 120,  72, 134,
        52,  65, 171, 107, 143,  32, 235, 114, 237,  23, 159,  86, 112,
       156, 231,  70, 127, 225, 169,  87, 181,  29, 155, 200,   7, 117,
       217,  80, 186,  61, 102, 211, 172,  41,  50,  67, 254,  10,  43,
       144, 128,  30,   5,  89, 227, 249, 213,  33, 228, 161, 119, 150,
       168,  40, 152, 178,  16, 133,  49, 242, 224, 108, 126,   3, 215,
        27, 204, 245, 184, 210, 212, 244,  48, 187,  13, 219,  91,  55,
       197,  11,  34, 137,  64, 202, 216,   0, 191,  12, 139, 149, 177,
       192,  20, 205,  92, 179,  42, 105, 232,   6,  77, 165,  62,  75,
       116, 226, 162, 136, 103, 214,  26, 129, 247], dtype=tf.int32)
    return perm, inv_perm
  elif b_type == 'gray':
    gray_code = GrayCode(8)
    return gray_code.codes, gray_code.inv_codes
  else:
    raise ValueError(f'Unknown b_type {b_type}')
# pylint: enable=bad-whitespace,bad-continuation


def get_x_channels(b_type):
  x_channels = 24 if b_type in ['uint8', 'uint8_s', 'gray'] else 9
  if b_type == 'oneh':
    x_channels = 256 * 3
  return x_channels


def rgb2bit(images, b_type, b_scale, x_channels):  # pylint: disable=missing-function-docstring
  if b_type in ['uint8', 'uint8_s', 'gray']:
    images = tf.image.convert_image_dtype(images, dtype=tf.uint8)
    if b_type in ['uint8_s', 'gray']:
      images = tf.gather(get_perm_inv_perm(b_type)[0],
                         tf.cast(images, tf.int32))
    images = utils.int2bits(
        tf.cast(images, tf.int32), x_channels // 3, tf.float32)
    sh = images.shape
    images = tf.reshape(images, sh[:-2] + [sh[-2] * sh[-1]])
    images = (images * 2 - 1) * b_scale
  elif b_type == 'oneh':
    images = tf.image.convert_image_dtype(images, dtype=tf.uint8)
    images = tf.one_hot(images, 256)
    sh = images.shape
    images = tf.reshape(images, sh[:-2] + [sh[-2] * sh[-1]])
    images = (images * 2 - 1) * b_scale
  else:
    raise ValueError(f'Unknown b_type {b_type}')
  return images


def bit2rgb(samples, b_type):  # pylint: disable=missing-function-docstring
  if b_type in ['uint8', 'uint8_s', 'gray']:
    samples = unfold_rgb(samples)
    samples = utils.bits2int(samples > 0, tf.int32)
    if b_type in ['uint8_s', 'gray']:
      samples = tf.gather(get_perm_inv_perm(b_type)[1], samples)
    return tf.image.convert_image_dtype(
        tf.cast(samples, tf.uint8), dtype=tf.float32)
  elif b_type == 'oneh':
    samples = unfold_rgb(samples)
    samples = tf.argmax(samples, -1)
    return tf.image.convert_image_dtype(
        tf.cast(samples, tf.uint8), dtype=tf.float32)
  else:
    raise ValueError(f'Unknown b_type {b_type}')


def unfold_rgb(x):
  # (b, h, w, d=3k) --> (b, h, w, 3, k) or (h, w, d=3k) --> (h, w, 3, k)
  if x.shape.rank <= 4:
    sh = utils.shape_as_list(x)
    x = tf.reshape(x, sh[:-1] + [3, sh[-1] // 3])
  return x


def fold_rgb(x):
  # (b, h, w, 3, k) --> (b, h, w, d=3k)
  if x.shape.rank == 5:
    sh = utils.shape_as_list(x)
    x = tf.reshape(x, sh[:-2] + [sh[-1] * sh[-2]])
  return x
