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
"""Modeling related utils."""

import math
import re
import tensorflow as tf
import tensorflow_addons as tfa


class WarmUpAndDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies a warmup schedule on a given learning rate decay schedule."""

  def __init__(self, base_learning_rate, learning_rate_scaling, batch_size,
               learning_rate_schedule, warmup_steps, total_steps,
               end_lr_factor=0.):
    super(WarmUpAndDecay, self).__init__()
    self.schedule = learning_rate_schedule
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.end_lr_factor = end_lr_factor
    if learning_rate_scaling == 'linear':
      self.base_lr = base_learning_rate * batch_size / 256.
    elif learning_rate_scaling == 'sqrt':
      self.base_lr = base_learning_rate * math.sqrt(batch_size)
    elif learning_rate_scaling == 'none' or learning_rate_scaling is None:
      self.base_lr = base_learning_rate
    else:
      raise ValueError('Unknown learning rate scaling {}'.format(
          learning_rate_scaling))

  def __call__(self, step):
    base_lr = self.base_lr
    schedule = self.schedule
    total_steps = self.total_steps
    warmup_steps = self.warmup_steps
    end_lr_factor = self.end_lr_factor

    if schedule == 'linear':
      linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
          base_lr, total_steps - warmup_steps,
          end_learning_rate=base_lr * end_lr_factor, power=1.0)
      decayed_lr = linear_decay(step - warmup_steps)
    elif schedule == 'cosine':
      cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
          base_lr, total_steps - warmup_steps, alpha=end_lr_factor)
      decayed_lr = cosine_decay(step - warmup_steps)
    elif schedule.startswith('exp@'):
      assert end_lr_factor == 0, 'non-zero end_lr not supported'
      rate = float(schedule.split('@')[1])
      exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(
          base_lr, total_steps - warmup_steps, rate)
      decayed_lr = exp_decay(step - warmup_steps)
    elif schedule == 'none':
      assert end_lr_factor == 0, 'non-zero end_lr not supported'
      decayed_lr = base_lr
    else:
      raise ValueError('Unknown learnig rate schedule {}'.format(
          schedule))

    learning_rate_warmup = (
        step / float(warmup_steps) * base_lr if warmup_steps else base_lr)
    learning_rate = tf.where(step < warmup_steps, learning_rate_warmup,
                             decayed_lr)
    return learning_rate


class AdamWeightDecay(tf.keras.optimizers.Adam):
  """Adam enables L2 weight decay and clip_by_global_norm on gradients.

  Just adding the square of the weights to the loss function is *not* the
  correct way of using L2 regularization/weight decay with Adam, since that will
  interact with the m and v parameters in strange ways.

  Instead we want ot decay the weights in a manner that doesn't interact with
  the m/v parameters. This is equivalent to adding the square of the weights to
  the loss with plain (non-momentum) SGD.
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               weight_decay_rate=0.0,
               include_in_weight_decay=None,
               exclude_from_weight_decay=None,
               name='AdamWeightDecay',
               **kwargs):
    super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2,
                                          epsilon, amsgrad, name, **kwargs)
    self.weight_decay_rate = weight_decay_rate
    self._include_in_weight_decay = include_in_weight_decay
    self._exclude_from_weight_decay = exclude_from_weight_decay

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype,
                                                apply_state)
    apply_state[(var_device, var_dtype)]['weight_decay_rate'] = tf.constant(
        self.weight_decay_rate, name='adam_weight_decay_rate')

  def _decay_weights_op(self, var, learning_rate, apply_state):
    do_decay = self._do_use_weight_decay(var.name)
    if do_decay:
      return var.assign_sub(
          learning_rate * var *
          apply_state[(var.device, var.dtype.base_dtype)]['weight_decay_rate'],
          use_locking=self._use_locking)
    return tf.no_op()

  def _get_lr(self, var_device, var_dtype, apply_state):
    """Retrieves the learning rate with the given state."""
    if apply_state is None:
      return self._decayed_lr_t[var_dtype], {}

    apply_state = apply_state or {}
    coefficients = apply_state.get((var_device, var_dtype))
    if coefficients is None:
      coefficients = self._fallback_apply_state(var_device, var_dtype)
      apply_state[(var_device, var_dtype)] = coefficients

    return coefficients['lr_t'], dict(apply_state=apply_state)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
    decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay,
                   self)._resource_apply_dense(grad, var, **kwargs)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
    decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay,
                   self)._resource_apply_sparse(grad, var, indices, **kwargs)

  def get_config(self):
    config = super(AdamWeightDecay, self).get_config()
    config.update({
        'weight_decay_rate': self.weight_decay_rate,
    })
    return config

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if self.weight_decay_rate == 0:
      return False

    if self._include_in_weight_decay:
      for r in self._include_in_weight_decay:
        if re.search(r, param_name) is not None:
          return True

    if self._exclude_from_weight_decay:
      for r in self._exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True


def build_optimizer(config, learning_rate):
  """Returns the optimizer."""
  if config.optimizer == 'momentum':
    return tf.keras.optimizers.SGD(
        learning_rate, config.momentum, nesterov=True)
  elif config.optimizer == 'adam':
    return tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=config.beta1,
        beta_2=config.beta2,
        epsilon=config.eps)
  elif config.optimizer == 'adamw':
    clipnorm = None if config.global_clipnorm <= 0 else config.global_clipnorm
    return AdamWeightDecay(
        learning_rate=learning_rate,
        weight_decay_rate=config.weight_decay,
        beta_1=config.beta1,
        beta_2=config.beta2,
        epsilon=config.eps,
        include_in_weight_decay=['kernel'],
        global_clipnorm=clipnorm)
  elif config.optimizer == 'lamb':
    return tfa.optimizers.LAMB(
        learning_rate=learning_rate,
        weight_decay_rate=config.weight_decay,
        beta_1=config.beta1,
        beta_2=config.beta2,
        epsilon=config.eps,
        exclude_from_weight_decay=['bias', 'beta', 'gamma', 'emb'])
  else:
    raise ValueError('Unknown optimizer {}'.format(config.optimizer))


def get_loss(logits, label_seq, loss_type):
  """Returns loss.

  Args:
    logits: tensor of shape (bsz, seqlen, vocab_size).
    label_seq: tensor of shape (bsz, seqlen).
    loss_type: string of loss type.

  Returns:
    per token loss tensor of shape (bsz, seqlen).
  """
  def _extract_loss_param(loss_type, default='0'):
    # loss_type is in `loss|loss@param` format where param is loss param.
    if '@' in loss_type:
      return loss_type.split('@')[1]
    return default

  label_hot = tf.cast(tf.one_hot(label_seq, tf.shape(logits)[-1]), logits.dtype)
  if 'xent' in loss_type:
    label_smoothing = float(_extract_loss_param(loss_type))
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=label_smoothing,
        reduction=tf.keras.losses.Reduction.NONE)(label_hot, logits)
  elif 'logistic' in loss_type:
    label_smoothing = float(_extract_loss_param(loss_type))
    logits -= tf.math.log(tf.cast(logits.shape[1], tf.float32))
    if label_smoothing > 0:
      label_hot = label_smoothing + label_hot * (1. - 2. * label_smoothing)
    log_p = tf.math.log_sigmoid(logits)
    log_p_not = tf.math.log_sigmoid(-logits)
    loss = -tf.reduce_sum(
        label_hot * log_p + (1. - label_hot) * log_p_not, axis=-1)
  elif 'focal' in loss_type:
    gamma = float(_extract_loss_param(loss_type))
    p = tf.nn.softmax(logits)
    logp = tf.math.log(p + 1e-8)
    focal_weight = tf.pow(1. - p, gamma) if gamma > 0 else 1.
    loss = - tf.reduce_sum(focal_weight * label_hot * logp, -1)
  else:
    raise ValueError('Unknown loss type {}'.format(loss_type))
  return loss


