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
"""The image diffusion model."""

import math
import ml_collections

import utils
from architectures.tape import ImageTapeDenoiser
from architectures.transformers import FITDenoiser
from architectures.transunet import TransUNet
from models import diffusion_utils
from models import model as model_lib
import tensorflow as tf


@model_lib.ModelRegistry.register('image_diffusion_model')
class Model(tf.keras.models.Model):
  """A model."""

  def __init__(self, config: ml_collections.ConfigDict, **kwargs):
    super().__init__(**kwargs)
    self.image_size = config.dataset.image_size
    self.num_classes = config.dataset.num_classes
    config = config.model
    self.config = config
    self.scheduler = diffusion_utils.Scheduler(config.train_schedule)
    if config.x0_clip == 'auto':
      self.x0_clip = '{},{}'.format(-config.b_scale, config.b_scale)
    else:
      self.x0_clip = config.x0_clip
    if config.arch_name == 'transunet':
      if ',' in config.kernel_sizes:
        kernel_sizes = [int(x) for x in config.kernel_sizes.split(',')]
      else:
        kernel_sizes = int(config.kernel_sizes)
      m_kwargs = dict(
          out_dim=self.sample_shape[-1],
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
          image_height=self.sample_shape[0],
          image_width=self.sample_shape[1],
          image_channels=self.sample_shape[-1],
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

  @property
  def sample_shape(self):
    return [self.image_size, self.image_size, 3]

  def get_cond_denoise(self, labels):
    config = self.config
    def cond_denoise(x, gamma, training, drop_label=False):
      gamma = tf.reshape(gamma, [-1])
      cond = None
      if config.conditional == 'class' or config.conditional == 'text':
        cond_dropout = config.get('cond_dropout', 0.)
        labels_w = 1.
        if training and cond_dropout > 0:
          if config.conditional == 'class':
            label_shape = [tf.shape(labels)[0], 1]
          else:
            label_shape = [tf.shape(labels)[0], 1, 1]
          labels_w = tf.random.uniform(label_shape) > cond_dropout
          labels_w = tf.cast(labels_w, tf.float32)
        if drop_label:
          labels_w = 0.
        if config.arch_name == 'transunet':  # Merge one-hot label with gamma.
          gamma = tf.concat([gamma[..., tf.newaxis], labels * labels_w], -1)
        else:
          cond = labels * labels_w
      elif config.conditional != 'none':
        raise ValueError(f'Unknown conditional {config.conditional}')
      return self.denoise(x, gamma, cond, training)
    return cond_denoise

  def denoise(self, x, gamma, cond, training):
    """gamma should be (bsz, ) or (bsz, d)."""
    assert gamma.shape.rank <= 2
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
    return x

  def sample(self, num_samples=100, iterations=100, method='ddim', **kwargs):
    config = self.config
    samples_shape = [num_samples, *self.sample_shape]
    if config.conditional == 'class':
      labels = tf.random.uniform(
          [num_samples], 0, self.num_classes, dtype=tf.int32)
      labels = tf.one_hot(labels, self.num_classes)
    elif config.conditional == 'text':
      labels = kwargs['labels']
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
    samples = (samples / config.b_scale / 2. + 0.5)  # convert -s,s -> 0,1

    if 'images' in kwargs and 'labels' in kwargs:
      images = self.images2tokens(kwargs['images'])
      images, noise, _, pred_dict = self.noise_denoise(
          images, kwargs['labels'], time_step=None, training=False)
      loss = self.compute_loss(images, noise, pred_dict)
    else:
      loss = tf.constant(-1.0, dtype=tf.float32)
    return samples, loss

  def noise_denoise(self, images, labels, time_step=None, training=True):
    config = self.config
    images = (images * 2. - 1.) * config.b_scale  # convert 0,1 -> -s,s
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
    cond_denoise = self.get_cond_denoise(labels)
    denoise_out = cond_denoise(denoise_inputs, gamma, training)
    if isinstance(denoise_out, tuple): denoise_out = denoise_out[0]
    x0_clip_fn = diffusion_utils.get_x0_clipping_function(self.x0_clip)
    pred_dict = diffusion_utils.get_x0_eps(
        images_noised, gamma, denoise_out, config.pred_type, x0_clip_fn,
        truncate_noise=False)
    return images, noise, images_noised, pred_dict

  def compute_loss(self,
                   images: tf.Tensor,
                   noise: tf.Tensor,
                   pred_dict: dict[str, tf.Tensor]) -> tf.Tensor:
    config = self.config
    loss_type = config.get('loss_type', config.pred_type)
    if loss_type == 'x':
      loss = tf.reduce_mean(
          tf.square((images - pred_dict['data_pred']) / config.b_scale))
    elif loss_type == 'eps':
      loss = tf.reduce_mean(tf.square(noise - pred_dict['noise_pred']))
    else:
      raise ValueError(f'Unknown pred_type {config.pred_type}')
    return loss

  def call(self,
           images: tf.Tensor,
           labels: tf.Tensor,
           training: bool = True,
           **kwargs)  -> tf.Tensor:  # pylint: disable=signature-mismatch
    """Model inference call."""
    with tf.name_scope(''):  # for other functions to have the same name scope
      images, noise, _, pred_dict = self.noise_denoise(
          images, labels, None, training)
      return self.compute_loss(images, noise, pred_dict)


@model_lib.TrainerRegistry.register('image_diffusion_model')
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


@model_lib.ModelRegistry.register('image_token_diffusion_model')
class ModelT(Model):
  """A model."""

  def __init__(self, config: ml_collections.ConfigDict, **kwargs):
    super(Model, self).__init__(**kwargs)
    self.image_size = config.dataset.image_size
    self.num_classes = config.dataset.num_classes
    config = config.model
    self.config = config
    self.scheduler = diffusion_utils.Scheduler(config.train_schedule)
    if config.x0_clip == 'auto':
      self.x0_clip = '{},{}'.format(-config.b_scale, config.b_scale)
    else:
      self.x0_clip = config.x0_clip
    self.num_groups = config.num_groups
    s_num_groups = int(math.sqrt(self.num_groups))
    self.sub_image_size = self.image_size // s_num_groups
    self.x_size = (self.image_size // config.patch_size)**2
    self.x_per_group = self.x_size // self.num_groups
    self.channels = config.patch_size**2 * 3
    if config.arch_name == 'fit':
      m_kwargs = dict(
          layers=config.num_layers,
          x_size=self.x_size,
          num_groups=config.num_groups,
          latents_per_group=config.latent_slots//config.num_groups,
          x_dim=config.tape_dim,
          latent_dim=config.latent_dim,
          x_num_heads=config.rw_num_heads,
          latent_num_heads=config.latent_num_heads,
          out_dim=self.channels,
          mlp_ratio=config.latent_mlp_ratio,
          drop_path=config.drop_path,
          drop_units=config.drop_units,
          drop_att=config.drop_att,
          x_pos_encoding=config.tape_pos_encoding,
          latent_pos_encoding=config.latent_pos_encoding,
          mask='none',
          cond_proj=True,
          self_cond=config.self_cond,
          x_self_attention=config.x_self_attention,
          xattn_with_mlp=config.xattn_with_mlp,
          xattn_enc_ln=config.xattn_enc_ln,
          name=config.arch_name)
      self.denoiser = FITDenoiser(**m_kwargs)
      self.denoiser_ema = FITDenoiser(**m_kwargs, trainable=False)
    else:
      raise ValueError(f'Unknown architecture {config.arch_name}')
    # Obtain hidden shapes for latent self conditioning.
    # TODO(iamtingchen): better way to handle it.
    self.hidden_shapes = getattr(self.denoiser, 'hidden_shapes', None)
    if self.hidden_shapes is not None:  # latent self-cond
      assert config.self_cond not in ['x', 'eps', 'auto']

  @property
  def sample_shape(self):
    return [self.num_groups, self.x_per_group, self.channels]

  def images2tokens(self, images):
    return utils.images2subimages2tokens(
        images,
        [self.sub_image_size, self.sub_image_size],
        [self.config.patch_size, self.config.patch_size])

  def tokens2images(self, tokens):
    return utils.tokens2subimages2images(
        tokens,
        [self.sub_image_size, self.sub_image_size],
        [self.config.patch_size, self.config.patch_size],
        [self.image_size, self.image_size])

  def sample(self, num_samples=100, iterations=100, method='ddim', **kwargs):
    samples, loss = super().sample(num_samples, iterations, method, **kwargs)
    return self.tokens2images(samples), loss

  def call(self,
           images: tf.Tensor,
           labels: tf.Tensor,
           training: bool = True,
           **kwargs)  -> tf.Tensor:  # pylint: disable=signature-mismatch
    """Model inference call."""
    with tf.name_scope(''):  # for other functions to have the same name scope
      images = self.images2tokens(images)
      images, noise, _, pred_dict = self.noise_denoise(
          images, labels, None, training)
      return self.compute_loss(images, noise, pred_dict)


model_lib.TrainerRegistry.register('image_token_diffusion_model')(Trainer)
