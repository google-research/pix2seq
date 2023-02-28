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
"""The model."""

import ml_collections

import utils
from architectures import convnet_blocks as convnets_lib
from architectures.transformers import get_shape
from architectures.transformers import ResNetTransformer
from architectures.transformers import VisionTransformer
from architectures.transunet import TransUNet
from models import diffusion_utils
from models import model as model_lib
import tensorflow as tf


@model_lib.ModelRegistry.register('panoptic_diffusion')
class Model(tf.keras.models.Model):
  """Inputs images and returns activations."""

  def __init__(self, config: ml_collections.ConfigDict, **kwargs):
    super().__init__(**kwargs)
    self.task_config = task_config = config.task
    self.decoder_config = decoder_config = config.decoder
    self.config = config = config.model
    assert config.conditional in ['cat', 'attn', 'cat+attn', 'none']
    encoder_trainable = False if (
        config.frozen_backbone or config.conditional == 'none') else True
    encoder_fuse_trainable = False if config.conditional == 'none' else True

    self.scheduler = diffusion_utils.Scheduler(config.train_schedule)
    if config.x0_clip == 'auto':
      self.x0_clip = '{},{}'.format(-config.b_scale, config.b_scale)
    else:
      self.x0_clip = config.x0_clip

    mlp_ratio = config.dim_mlp // config.dim_att
    if config.resnet_variant == 'c1':
      self.encoder = VisionTransformer(
          config.image_size[0], config.image_size[1], config.patch_size,
          config.num_encoder_layers, config.dim_att, mlp_ratio,
          config.num_heads, config.drop_path, config.drop_units,
          config.drop_att, config.pos_encoding, config.use_cls_token,
          name='vit', trainable=encoder_trainable)
      self.encoder_ema = VisionTransformer(
          config.image_size[0], config.image_size[1], config.patch_size,
          config.num_encoder_layers, config.dim_att, mlp_ratio,
          config.num_heads, config.drop_path, config.drop_units,
          config.drop_att, config.pos_encoding, config.use_cls_token,
          name='vit', trainable=False)
    else:
      self.encoder = ResNetTransformer(
          config.image_size[0], config.image_size[1], config.resnet_variant,
          config.resnet_depth, config.resnet_width_multiplier,
          config.resnet_sk_ratio, config.num_encoder_layers, config.dim_att,
          mlp_ratio, config.num_heads, config.drop_path, config.drop_units,
          config.drop_att, config.pos_encoding, config.use_cls_token,
          name='rest', trainable=encoder_trainable)
      self.encoder_ema = ResNetTransformer(
          config.image_size[0], config.image_size[1], config.resnet_variant,
          config.resnet_depth, config.resnet_width_multiplier,
          config.resnet_sk_ratio, config.num_encoder_layers, config.dim_att,
          mlp_ratio, config.num_heads, config.drop_path, config.drop_units,
          config.drop_att, config.pos_encoding, config.use_cls_token,
          name='rest', trainable=False)

    if config.enc_fuse == 'pyramid_merge':
      fuse_class = convnets_lib.FeaturePyramidMerge
    elif config.enc_fuse == 'pyramid_merge_naive':
      fuse_class = convnets_lib.FeaturePyramidMergeNaive
    else:
      raise ValueError(f'Unknown enc_fuse {config.enc_fuse}')
    m_kwargs = dict(
        out_dim=config.enc_fuse_dim,
        out_size=config.msize,
        upsample=config.enc_fuse_upsample,
        name='encoder_fuse')
    self.encoder_fuse = fuse_class(**m_kwargs, trainable=encoder_fuse_trainable)
    self.encoder_fuse_ema = fuse_class(**m_kwargs, trainable=False)

    self.x_channels = task_config.n_bits_label
    if ',' in decoder_config.kernel_sizes:
      kernel_sizes = [int(x) for x in decoder_config.kernel_sizes.split(',')]
    else:
      kernel_sizes = int(decoder_config.kernel_sizes)
    n_res_blocks = [int(x) for x in decoder_config.n_res_blocks.split(',')]
    ch_multi = [int(x) for x in decoder_config.ch_multipliers.split(',')]
    mhsa_res = [int(x) for x in decoder_config.mhsa_resolutions.split(',')]
    m_kwargs = dict(
        out_dim=self.x_channels,
        dim=decoder_config.dim,
        in_strides=decoder_config.in_strides,
        in_kernel_size=decoder_config.in_kernel_size,
        out_kernel_size=decoder_config.out_kernel_size,
        n_mlp_blocks=decoder_config.n_mlp_blocks,
        n_res_blocks=n_res_blocks,
        kernel_sizes=kernel_sizes,
        ch_multipliers=ch_multi,
        dropout=decoder_config.udrop,
        mhsa_resolutions=mhsa_res,
        per_head_dim=decoder_config.per_head_dim,
        transformer_dim=decoder_config.transformer_dim,
        transformer_strides=decoder_config.transformer_strides,
        transformer_blocks=decoder_config.transformer_blocks,
        conditioning=('attn' in config.conditional),
        time_scaling=config.time_scaling,
        pos_encoding=decoder_config.u_pos_encoding,
        norm_type='group_norm',
        outp_softmax_groups=decoder_config.outp_softmax_groups,
        b_scale=config.b_scale,
        name='denoiser')
    self.denoiser = TransUNet(**m_kwargs)
    self.denoiser_ema = TransUNet(**m_kwargs, trainable=False)

    # TODO(iamtingchen): better way to handle it.
    self.hidden_shapes = getattr(self.denoiser, 'hidden_shapes', None)
    if self.hidden_shapes is not None:  # latent self-cond
      assert config.self_cond not in ['x', 'eps', 'auto']

  def encode_images_null(self, images, training):
    config = self.config
    if not hasattr(self, 'images_shape'):
      self.images_shape = tf.shape(images)
    encoded_cat = tf.zeros([tf.shape(images)[0], config.msize[0],
                            config.msize[1],
                            config.enc_fuse_dim])
    return encoded_cat, None

  def encode_images(self, images, training):
    """Encode images into latents for decoder to condition on."""
    config = self.config
    if not hasattr(self, 'images_shape'):
      self.images_shape = tf.shape(images)
    encoder = self.encoder if training else self.encoder_ema
    encoder_fuse = self.encoder_fuse if training else self.encoder_fuse_ema
    if config.resnet_variant == 'c1':
      encoded = encoder(images, training)
      bsz, seqlen, _ = get_shape(encoded)
      hf_ = wf_ = tf.cast(tf.math.sqrt(tf.cast(seqlen, tf.float32)), tf.int32)
      features = [tf.reshape(encoded, [bsz, hf_, wf_, tf.shape(encoded)[-1]])]
    else:
      encoded, encoded_list = encoder(images, training, ret_list=True)
      bsz, hf_, wf_, _ = get_shape(encoded_list[-1])
      features = encoded_list[:-1] + [
          tf.reshape(encoded, [bsz, hf_, wf_, tf.shape(encoded)[-1]])]
    if config.frozen_backbone:
      encoded = tf.stop_gradient(encoded)
      features = [tf.stop_gradient(f) for f in features]
    encoded_cat = encoder_fuse(features, training)
    if 'cat' not in config.conditional:
      encoded_cat *= 0  # keep vars and shape the same.
    return encoded_cat, encoded

  def get_cond_denoise(self, encoded_cat, encoded, cond_map=None,
                       return_logits=False):
    def cond_denoise(samples, gamma, training):
      gamma = tf.reshape(gamma, [-1])
      if self.config.normalize_noisy_input:
        # rescaling vs normalization
        # gamma_ = tf.reshape(gamma, [tf.shape(gamma)[0], 1, 1, 1])
        # samples /= tf.sqrt((self.config.b_scale**2-1) * gamma_ + 1)
        samples /= tf.math.reduce_std(
            samples, list(range(1, samples.shape.ndims)), keepdims=True)
      if encoded_cat is not None:
        if isinstance(samples, tuple) or isinstance(samples, list):
          samples = list(samples)
          samples[0] = tf.concat([samples[0], encoded_cat], -1)
        else:
          samples = tf.concat([samples, encoded_cat], -1)
      if cond_map is not None:
        samples = tf.concat([samples, cond_map], -1)
      return self.denoise(samples, encoded, gamma, training, return_logits)
    return cond_denoise

  def denoise(self, x, c, gamma, training, return_logits=False):
    config = self.config
    if not hasattr(self, 'denoise_x_shape'):
      if isinstance(x, tuple) or isinstance(x, list):
        self.denoise_x_shape = tuple(tf.shape(x_) for x_ in x)
      else:
        self.denoise_x_shape = tf.shape(x)
      if c is not None:
        self.denoise_c_shape = tf.shape(c)
      self.denoise_gamma_shape = tf.shape(gamma)
    denoiser = self.denoiser if training else self.denoiser_ema
    x = denoiser(x, gamma, c, training, return_logits)
    if config.pred_type == 'x_sigmoid_xent':
      x = x if training else (tf.nn.sigmoid(x) * 2 - 1) * config.b_scale
    return x

  def infer(self, images, iterations=100, method='ddim'):
    config = self.config
    if isinstance(images, tuple):
      images, cond_map = images
    else:
      cond_map = None
    samples_shape = [tf.shape(images)[0], config.msize[0],
                     config.msize[1], self.x_channels]
    if config.conditional == 'none':
      encoded_cat, encoded = self.encode_images_null(images, training=False)
    else:
      encoded_cat, encoded = self.encode_images(images, training=False)
    masks = self.scheduler.generate(
        self.get_cond_denoise(encoded_cat, encoded, cond_map),
        iterations,
        samples_shape,
        hidden_shapes=self.hidden_shapes,
        pred_type=config.pred_type,
        schedule=config.infer_schedule,
        td=config.td,
        x0_clip=self.x0_clip,
        self_cond=config.self_cond,
        sampler_name=method)
    return masks

  def noise_denoise(self, images, masks, time_step=None, training=True):
    config = self.config
    if isinstance(images, tuple):
      images, cond_map = images
    else:
      cond_map = None
    if config.conditional == 'none':
      encoded_cat, encoded = self.encode_images_null(images, training=training)
    else:
      encoded_cat, encoded = self.encode_images(images, training=training)
    if config.l_tile_factors > 1:
      encoded_cat = utils.tile_along_batch(encoded_cat, config.l_tile_factors)
      encoded = utils.tile_along_batch(encoded, config.l_tile_factors)
      masks = utils.tile_along_batch(masks, config.l_tile_factors)
    masks_noised, noise, _, gamma = self.scheduler.add_noise(
        masks, time_step=time_step)
    if config.self_cond != 'none':
      sc_rate = config.get('self_cond_rate', 0.5)
      self_cond_by_masking = config.get('self_cond_by_masking', False)
      if self_cond_by_masking:
        sc_drop_rate = 1. - sc_rate
        num_sc_data = tf.shape(images)[0]
      else:
        sc_drop_rate = 0.
        num_sc_data = tf.cast(
            tf.cast(tf.shape(images)[0], tf.float32) * sc_rate, tf.int32)
      encoded_cat_p = None if encoded_cat is None else encoded_cat[:num_sc_data]
      encoded_p = None if encoded is None else encoded[:num_sc_data]
      cond_map_p = None if cond_map is None else cond_map[:num_sc_data]
      cond_denoise = self.get_cond_denoise(encoded_cat_p, encoded_p, cond_map_p)
      if self.hidden_shapes is None:  # data self-cond, return is a tensor.
        denoise_inputs = diffusion_utils.add_self_cond_estimate(
            masks_noised, gamma, cond_denoise, config.pred_type,
            config.self_cond, self.x0_clip, num_sc_data,
            drop_rate=sc_drop_rate, training=training)
      else:  # latent self-cond, return is a tuple.
        denoise_inputs = diffusion_utils.add_self_cond_hidden(
            masks_noised, gamma, cond_denoise, num_sc_data,
            self.hidden_shapes, drop_rate=sc_drop_rate, training=training)
    else:
      denoise_inputs = masks_noised
    cond_denoise = self.get_cond_denoise(
        encoded_cat, encoded, cond_map, return_logits=True)
    denoise_out = cond_denoise(denoise_inputs, gamma, training)
    if isinstance(denoise_out, tuple): denoise_out = denoise_out[0]
    return masks, noise, masks_noised, denoise_out

  def compute_loss(self, masks, noise, denoise_out, masks_weight):
    config = self.config
    if config.l_tile_factors > 1:
      masks_weight = utils.tile_along_batch(masks_weight, config.l_tile_factors)
    masks_weight = masks_weight[..., tf.newaxis]  # (b, h, w) --> (b, h, w, 1)
    if config.pred_type == 'x':
      loss = tf.reduce_mean(tf.square(masks - denoise_out) * masks_weight)
    elif config.pred_type == 'x_sigmoid_xent':
      pp = tf.nn.sigmoid(denoise_out * masks / config.b_scale)
      loss = tf.reduce_mean(-tf.math.log(pp + 1e-8) * masks_weight)
    elif config.pred_type == 'x_softmax_xent':
      # TODO(iamtingchen): avoid re-computing the one-hot targets.
      xs = tf.split(masks, self.decoder_config.outp_softmax_groups, -1)
      targets = utils.bits2int(tf.stack(xs, -2) > 0, tf.int32)
      targets = tf.one_hot(targets, tf.shape(denoise_out)[-1])
      loss = tf.reduce_mean(
          masks_weight *
          tf.nn.softmax_cross_entropy_with_logits(targets, denoise_out))
    elif config.pred_type == 'eps':
      loss = tf.reduce_mean(tf.square(noise - denoise_out))
    else:
      raise ValueError(f'Unknown pred_type {config.pred_type}')
    return loss

  def call(self, images, masks, masks_weight, training):
    """Model inference call."""
    with tf.name_scope(''):  # for other functions to have the same name scope.
      masks, noise, _, denoise_out = self.noise_denoise(
          images, masks, None, training)
      return self.compute_loss(masks, noise, denoise_out, masks_weight)


@model_lib.TrainerRegistry.register('panoptic_diffusion')
class Trainer(model_lib.Trainer):
  """A trainer."""

  def compute_loss(self, preprocess_outputs):
    """Compute loss based on model outputs and targets."""
    images, masks, masks_weight = preprocess_outputs
    loss = self.model(images, masks, masks_weight, training=True)
    return loss

  def train_step(self, examples, tasks, strategy):
    super().train_step(examples, tasks, strategy)

    # EMA udpate
    oconfig = self.config.optimization
    ema_decay = oconfig.get('ema_decay', 0.)
    if self.config.model.conditional == 'none' and not hasattr(
        self.model, 'encoder_initialized'):
      self.model.encoder_initialized = True
      _ = self.model.encode_images(
          tf.zeros(self.model.images_shape), training=True)
    vars_src = (self.model.denoiser.variables +
                self.model.encoder.variables +
                self.model.encoder_fuse.variables)
    if not hasattr(self.model, 'ema_initialized'):
      self.model.ema_initialized = True
      _ = self.model.encode_images(
          tf.zeros(self.model.images_shape), training=False)
      if isinstance(self.model.denoise_x_shape, tuple):
        x = tuple(tf.zeros(sh) for sh in self.model.denoise_x_shape)
      else:
        x = tf.zeros(self.model.denoise_x_shape)
      _ = self.model.denoise(
          x=x,
          c=tf.zeros(self.model.denoise_c_shape) if hasattr(
              self.model, 'denoise_c_shape') else None,
          gamma=tf.zeros(self.model.denoise_gamma_shape),
          training=False)
    vars_dst = (self.model.denoiser_ema.variables +
                self.model.encoder_ema.variables +
                self.model.encoder_fuse_ema.variables)
    assert len(vars_src) == len(vars_dst), (len(vars_src), len(vars_dst))
    if oconfig.get('ema_name_exact_match', False):
      src_vars_dict = dict((var.name, var) for var in vars_src)
      vars_src = [src_vars_dict[var.name] for var in vars_dst]
    for var_src, var_dst in zip(vars_src, vars_dst):
      var_dst.assign(var_dst * ema_decay + var_src * (1. - ema_decay))
