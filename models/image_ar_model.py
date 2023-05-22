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
"""The image autoregressive decoder model."""

import einops
import ml_collections

import utils
from architectures.transformers import AutoregressiveDecoder
from architectures.transformers import FITAR
from models import model as model_lib
from models import model_utils
import tensorflow as tf


@model_lib.ModelRegistry.register('image_ar_decoder')
class Model(tf.keras.models.Model):
  """Inputs images and returns activations."""

  def __init__(self, config: ml_collections.ConfigDict, **kwargs):
    # vocab_size and max_seq_len don't include start token, which is only used
    # inside this class.
    super().__init__(**kwargs)
    image_size = config.dataset.image_size
    self.loss_type = config.train.loss_type
    config = config.model
    self.config = config
    if config.arch_name == 'base':
      mlp_ratio_dec = config.dim_mlp_dec // config.dim_att_dec
      self.decoder = AutoregressiveDecoder(
          config.vocab_size, config.max_seq_len, config.num_decoder_layers,
          config.dim_att_dec, mlp_ratio_dec, config.num_heads_dec,
          config.drop_path, config.drop_units, config.drop_att,
          config.pos_encoding_dec, config.shared_decoder_embedding,
          config.decoder_output_bias, cross_attention=False, name='ar_decoder')
    else:
      self.decoder = FITAR(
          layers=config.layers,
          x_size=image_size**2*3,
          num_groups=(image_size//config.patch_size)**2,
          latents_per_group=config.latents_per_group,
          x_dim=config.dim_att,
          latent_dim=config.dim_latent,
          x_num_heads=config.num_heads,
          latent_num_heads=config.num_heads,
          mlp_ratio=config.dim_mlp//config.dim_att,
          vocab_size=config.vocab_size,
          shared_embedding=config.shared_decoder_embedding,
          output_bias=config.decoder_output_bias,
          drop_path=config.drop_path,
          drop_units=config.drop_units,
          drop_att=config.drop_att,
          x_pos_encoding=config.pos_encoding,
          latent_pos_encoding=config.latent_pos_encoding)

  def call(self, images, labels=None, training=True):
    """Model function call for *training*."""
    with tf.name_scope(''):  # for other functions to have the same name scope.
      config = self.config
      input_seq, target_seq = image2seqs(
          images, config.arch_name, config.patch_size, config.patch_ordering)
      logits = self.decoder(input_seq, None, training=training)
      losses = model_utils.get_loss(logits, target_seq, self.loss_type)
      loss = tf.reduce_mean(losses) / tf.math.log(2.0)
      return loss, logits, target_seq

  def sample(self, **kwargs):
    """Sampling."""
    # TODO(iamtingchen): add sampling.
    loss, _, _ = self.call(kwargs['images'], kwargs['labels'], training=False)
    return kwargs['images'], loss


@model_lib.TrainerRegistry.register('image_ar_decoder')
class ARTrainer(model_lib.Trainer):
  """A trainer for AR model."""

  def __init__(self, config: ml_collections.ConfigDict, **kwargs):
    """Init and setup basic training elements under strategy scope.

    Note: the trainer needs to be created under `strategy.scope()`.

    Args:
      config: object for holding hyperparameters and other configurations.
      **kwargs: other neccesary configurations to pass for training setup.
    """
    super().__init__(config, **kwargs)
    self._metrics.update({
        'loss': tf.keras.metrics.Mean('loss'),
        'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(
            'accuracy'),
    })

  def compute_loss(self, preprocess_outputs):
    """Compute loss based on model outputs and targets."""
    images, labels = preprocess_outputs
    loss, logits, target_seq = self.model(images, labels, training=True)

    # update metrics
    self._metrics['loss'].update_state(loss)
    self._metrics['accuracy'].update_state(target_seq, logits)

    return loss


def image2seqs(images, arch_name, patch_size, patch_ordering='snake'):
  """Turn images into input and target sequences."""
  if arch_name == 'base':
    images = einops.rearrange(images, 'b h w c -> b (h w c)')
    target_seq = tf.cast(images * 255., tf.int32)  # (bsz, seqlen)
    input_seq = tf.concat(
        [tf.zeros_like(target_seq[:, :1]), target_seq[:, :-1]], 1)
  else:
    images = utils.extract_patches(
        images, [patch_size, patch_size], patch_ordering=patch_ordering)
    target_seq = tf.cast(images * 255., tf.int32)  # (bsz, groups, seqlen)
    flat_seq = einops.rearrange(target_seq, 'b n m -> b (n m)')
    input_seq = tf.concat(
        [tf.zeros_like(flat_seq[:, :1]), flat_seq[:, :-1]], 1)
    input_seq = tf.reshape(input_seq, tf.shape(target_seq))
  return input_seq, target_seq
