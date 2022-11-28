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
"""The image encoder and autoregressive decoder model."""

import ml_collections

import utils
from architectures.transformers import add_vis_pos_emb
from architectures.transformers import AutoregressiveDecoder
from architectures.transformers import MLP
from architectures.transformers import ResNetTransformer
from architectures.transformers import VisionTransformer
from models import model as model_lib
from models import model_utils
import tensorflow as tf


@model_lib.ModelRegistry.register('encoder_ar_decoder')
class Model(tf.keras.models.Model):
  """Inputs images and returns activations."""

  def __init__(self, config: ml_collections.ConfigDict, **kwargs):
    # vocab_size and max_seq_len don't include start token, which is only used
    # inside this class.
    super().__init__(**kwargs)
    config = config.model
    self.config = config

    mlp_ratio = config.dim_mlp // config.dim_att
    if config.resnet_variant == 'c1':
      self.encoder = VisionTransformer(
          config.image_size, config.image_size, config.patch_size,
          config.num_encoder_layers, config.dim_att, mlp_ratio,
          config.num_heads, config.drop_path, config.drop_units,
          config.drop_att, config.pos_encoding, config.use_cls_token,
          name='vit')
    else:
      self.encoder = ResNetTransformer(
          config.image_size, config.image_size, config.resnet_variant,
          config.resnet_depth, config.resnet_width_multiplier,
          config.resnet_sk_ratio, config.num_encoder_layers, config.dim_att,
          mlp_ratio, config.num_heads, config.drop_path, config.drop_units,
          config.drop_att, config.pos_encoding, config.use_cls_token,
          name='rest')

    mlp_ratio_dec = config.dim_mlp_dec // config.dim_att_dec
    self.proj = tf.keras.layers.Dense(
        config.dim_att_dec, name='proj/linear')
    self.proj_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='proj/ln')
    if config.dec_proj_mode in ['linear_p', 'mlp']:
      add_vis_pos_emb(
          self, config.pos_encoding, self.encoder.n_rows, self.encoder.n_cols,
          config.dim_att_dec, name_prefix='proj')
      if config.dec_proj_mode == 'mlp':
        self.proj_mlp = MLP(1, config.dim_att_dec, mlp_ratio, config.drop_path,
                            config.drop_units, name='proj/mlp')

    self.decoder = AutoregressiveDecoder(
        config.vocab_size, config.max_seq_len, config.num_decoder_layers,
        config.dim_att_dec, mlp_ratio_dec, config.num_heads_dec,
        config.drop_path, config.drop_units, config.drop_att,
        config.pos_encoding_dec, config.shared_decoder_embedding,
        config.decoder_output_bias, name='ar_decoder')

  def _tile_vis_output(self, vis_output, seq):
    """Tile vis_output per seq.

    Args:
      vis_output: `float` tensor of encoded images in shape of (bsz, ....).
      seq: `int` sequence in shape of (bsz, seqlen),
        or (bsz, instances, seqlen) if there are multiple sequences per image.

    Returns:
      vis_output of (bsz*instances, ...).
      seq of (bsz*instances, ...).
    """
    if seq.shape.rank > 2:
      tile_factor = seq.shape.as_list()[-2]
      vis_output = utils.tile_along_batch(vis_output, tile_factor)
      seq = utils.flatten_batch_dims(seq, out_rank=2)
    return vis_output, seq

  def _encode_images(self, images, training):
    """Encode images into latents for decoder to condition on."""
    config = self.config
    encoded = self.encoder(images, training)
    encoded = self.proj_ln(self.proj(encoded))
    # Add (optional) positional embedding to encoded visual units.
    if config.dec_proj_mode != 'linear':
      vis_pos_emb = tf.expand_dims(self.vis_pos_emb, 0)
      if config.use_cls_token:
        encoded = encoded + tf.concat(
            [tf.zeros_like(vis_pos_emb[:, :1]), vis_pos_emb], 1)
      else:
        encoded = encoded + vis_pos_emb
      if config.dec_proj_mode == 'mlp':
        encoded = self.proj_mlp(encoded, training)
      else:
        assert config.dec_proj_mode == 'linear_p'
    return encoded

  def call(self, images, seq, training=True):
    """Model function call for *training*.

    Args:
      images: `float` tensor of (bsz, h, w, c).
      seq: `int` sequence visible to the model of shape (bsz, seqlen),
        or (bsz, instances, seqlen) if there are multiple sequences per image.
      training: `bool` indicator.

    Returns:
      logits for each predicted tokens of (bsz * instances, seqlen, vocab_size).
    """
    with tf.name_scope(''):  # for other functions to have the same name scope.
      encoded = self._encode_images(images, training)
      encoded, seq = self._tile_vis_output(encoded, seq)
      logits = self.decoder(seq, encoded, training)
      return logits

  def infer(self, images, prompt_seq, encoded=None, max_seq_len=None,
            temperature=1, top_k=1, top_p=1., num_samples=1,
            sampling_callback=None):
    """Model function call for inference.

    Args:
      images: `float` tensor of (bsz, h, w, c).
      prompt_seq: `int` sequence visible to the model of shape (bsz, seqlen),
        or (bsz, instances, seqlen) if there are multiple sequences per image.
      encoded: cache for encoded images for decoder. Skip image encoding if this
        is given.
      max_seq_len: `int` of max generated sequence length (including prompt).
      temperature: `float` scalar for scaling the logits before sampling.
      top_k: `int` scalar for truncating top-k tokens according to logits before
        token sampling.
      top_p: `float` scalar specifying the threshold of cumulative probablity
        for truncating tokens before token sampling.
      num_samples: `int` number of samples to be generated for each instance.
      sampling_callback: a callbak `function` that take `next_logits`, and
        return `next_token`. This is used when users need a specific logic
        for sampling. Default to `None` with standard free-form sampling.

    Returns:
      pred_seq: `int` prediction sequence of shape
          (bsz * instances * num_samples, seqlen)
      logits: `float` of shape
          (bsz * instances * num_samples, seqlen, vocab_size)
      encoded: `float` tensor of encoded images.
    """
    if encoded is None:
      encoded = self._encode_images(images, training=False)
    encoded, prompt_seq = self._tile_vis_output(encoded, prompt_seq)

    # Tile by num_samples too.
    encoded = utils.tile_along_batch(encoded, num_samples)
    prompt_seq = utils.tile_along_batch(prompt_seq, num_samples)

    pred_seq, logits = self.decoder.infer(
        prompt_seq, encoded, max_seq_len,
        temperature, top_k, top_p, sampling_callback)

    return pred_seq, logits, encoded


@model_lib.TrainerRegistry.register('encoder_ar_decoder')
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
        'loss_notpad': tf.keras.metrics.Mean('loss_notpad'),
        'accuracy_notpad': tf.keras.metrics.SparseCategoricalAccuracy(
            'accuracy_notpad'),
    })

  def compute_loss(self, preprocess_outputs):
    """Compute loss based on model outputs and targets."""
    image, input_seq, target_seq, token_weights = preprocess_outputs

    target_seq = utils.flatten_batch_dims(target_seq, out_rank=2)
    token_weights = utils.flatten_batch_dims(token_weights, out_rank=2)
    token_weights = utils.tf_float32(token_weights)
    is_padding = tf.equal(target_seq, 0)  # padding tokens.
    token_weights_notpad = tf.where(
        is_padding, tf.zeros_like(token_weights), token_weights)

    logits = self.model(image, input_seq)
    losses = model_utils.get_loss(
        logits, target_seq, self.config.train.loss_type)
    loss = tf.reduce_sum(losses * token_weights) / (
        tf.reduce_sum(token_weights) + 1e-9)
    loss_notpad = tf.reduce_sum(losses * token_weights_notpad) / (
        tf.reduce_sum(token_weights_notpad) + 1e-9)

    # update metrics
    self._metrics['loss_notpad'].update_state(loss_notpad)
    self._metrics['accuracy_notpad'].update_state(
        tf.boolean_mask(target_seq, tf.greater(token_weights_notpad, 0)),
        tf.boolean_mask(logits, tf.greater(token_weights_notpad, 0)))

    return loss
