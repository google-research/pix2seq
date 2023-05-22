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

import functools
import ml_collections

from architectures.tape import VideoTapeDenoiser
from models import diffusion_utils
from models import image_diffusion_model
from models import model as model_lib
from models.diffusion_utils import Scheduler
import tensorflow as tf


@model_lib.ModelRegistry.register('video_diffusion_model')
class Model(image_diffusion_model.Model):
  """A model with tape for video prediction."""

  def __init__(self, config: ml_collections.ConfigDict, **kwargs):
    super(image_diffusion_model.Model, self).__init__(**kwargs)
    image_size = config.dataset.image_size
    self.image_size = image_size
    self.num_classes = config.dataset.num_classes
    self.seq_len = config.dataset.seq_len
    config = config.model
    self.config = config
    self.scheduler = Scheduler(config.train_schedule)
    if config.x0_clip == 'auto':
      self.x0_clip = '{},{}'.format(-config.b_scale, config.b_scale)
    else:
      self.x0_clip = config.x0_clip
    self.x_channels = 3

    model_fn = functools.partial(
        VideoTapeDenoiser,
        num_layers=config.num_layers,
        latent_slots=config.latent_slots,
        latent_dim=config.latent_dim,
        latent_mlp_ratio=config.latent_mlp_ratio,
        latent_num_heads=config.latent_num_heads,
        tape_dim=config.tape_dim,
        tape_mlp_ratio=config.tape_mlp_ratio,
        rw_num_heads=config.rw_num_heads,
        image_height=image_size,
        image_width=image_size,
        image_channels=self.x_channels,
        patch_size=config.patch_size,
        seq_len=self.seq_len,
        seq_stride=config.seq_stride,
        seq_cond=self.seq_len - self.sample_shape[0],
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
        xattn_enc_ln=config.xattn_enc_ln,
        name=config.arch_name)
    self.denoiser = model_fn(name='denoiser')
    self.denoiser_ema = model_fn(name='denoiser', trainable=False)
    self.hidden_shapes = self.denoiser.hidden_shapes

  @property
  def sample_shape(self):
    if self.seq_len > 1:
      seq_cond = self.config.get('conditional', 'seq@0').split('@')
      seq_cond = int(seq_cond[-1]) if len(seq_cond) > 1 else 0
      return [self.seq_len-seq_cond, self.image_size, self.image_size,
              self.x_channels]
    else:
      return [self.image_size, self.image_size, self.x_channels]

  # override for conditional data
  def get_cond_denoise(self, labels, cond=None):
    config = self.config
    def cond_denoise(x, gamma, training):
      gamma = tf.reshape(gamma, [-1])
      if config.conditional == 'class':
        gamma = tf.concat([gamma[..., tf.newaxis], labels], -1)
      elif config.conditional != 'none' and 'seq@' not in config.conditional:
        raise ValueError(f'Unknown conditional {config.conditional}')
      return self.denoise(x, gamma, cond, training)
    return cond_denoise

  # override to pass x_cond
  def sample(self, num_samples=100, iterations=100, method='ddim', **kwargs):
    config = self.config
    samples_shape = [num_samples, *self.sample_shape]
    if config.conditional == 'class':
      labels = tf.random.uniform(
          [num_samples], 0, self.num_classes, dtype=tf.int32)
      labels = tf.one_hot(labels, self.num_classes)
    else:
      labels = None
    x_cond = kwargs['images'][:, :-self.sample_shape[0]]
    x_cond = (x_cond * 2. - 1.) * config.b_scale  # convert 0,1 -> -s,s
    samples = self.scheduler.generate(
        self.get_cond_denoise(labels, cond=x_cond),
        iterations,
        samples_shape,
        hidden_shapes=self.hidden_shapes,
        pred_type=config.pred_type,
        schedule=config.infer_schedule,
        td=config.td,
        noise_std=config.noise_std,
        x0_clip=self.x0_clip,
        self_cond=config.self_cond,
        sampler_name=config.sampler_name)
    if x_cond.shape[1] > 0:
      samples = tf.concat([x_cond, samples], axis=1)
    samples = (samples / config.b_scale / 2. + 0.5)  # convert -s,s -> 0,1
    return samples

  # override to allow for more cond data
  def noise_denoise(self, images, labels, time_step=None, training=True):
    config = self.config
    images = (images * 2. - 1.) * config.b_scale  # convert 0,1 -> -s,s
    seq_len = self.sample_shape[0]
    cond_images, images = images[:, :-seq_len], images[:, -seq_len:]
    images_noised, noise, _, gamma = self.scheduler.add_noise(
        images, time_step=time_step)
    images_noised_ori = images_noised
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
      cond_denoise = self.get_cond_denoise(
          labels[:num_sc_examples],
          cond_images[:num_sc_examples] if cond_images is not None else None)
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
    cond_denoise = self.get_cond_denoise(labels, cond_images)
    denoise_out = cond_denoise(denoise_inputs, gamma, training)
    if isinstance(denoise_out, tuple): denoise_out = denoise_out[0]

    return images, noise, images_noised_ori, denoise_out


@model_lib.TrainerRegistry.register('video_diffusion_model')
class Trainer(image_diffusion_model.Trainer):
  """A trainer."""

  def train_step(self, examples, tasks, strategy):
    super().train_step(examples, tasks, strategy)
