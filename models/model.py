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
"""Abstract model file."""

import abc
from absl import logging
import ml_collections
import registry
import utils
from models import model_utils
import tensorflow as tf

ModelRegistry = registry.Registry()
TrainerRegistry = registry.Registry()


class Trainer(abc.ABC):
  """A base trainer."""

  def __init__(self, config: ml_collections.ConfigDict, **kwargs):
    """Init and setup basic training elements under strategy scope.

    Note: the trainer needs to be created under `strategy.scope()`.

    Args:
      config: object for holding hyperparameters and other configurations.
      **kwargs: other neccesary configurations to pass for training setup.
    """
    self._config = config

    # Setup learning rate and optimizer.
    num_train_examples = kwargs['num_train_examples']
    train_steps = kwargs['train_steps']
    c_opt = config.optimization
    batch_size = config.train.batch_size
    end_lr_factor = c_opt.end_lr_factor if 'end_lr_factor' in c_opt else 0.
    warmup_steps = c_opt.warmup_steps or int(
        round(c_opt.warmup_epochs * num_train_examples // batch_size))
    self._learning_rate = learning_rate = model_utils.WarmUpAndDecay(
        c_opt.learning_rate, c_opt.learning_rate_scaling, batch_size,
        c_opt.learning_rate_schedule, warmup_steps, train_steps, end_lr_factor)
    self._optimizer = optimizer = model_utils.build_optimizer(
        config.optimization, learning_rate)

    # Setup model and checkpoints.
    self._model = model = ModelRegistry.lookup(config.model.name)(config)
    model_dir = kwargs['model_dir']
    latest_ckpt, ckpt, self._verify_restored = utils.restore_from_checkpoint(
        model_dir, False,
        model=model, global_step=optimizer.iterations, optimizer=optimizer)
    self._verify_restored_p = None
    if not latest_ckpt:
      if config.model.pretrained_ckpt:
        _, _, self._verify_restored_p = utils.restore_from_checkpoint(
            config.model.pretrained_ckpt, True, model=model)
    self._checkpoint_manager = tf.train.CheckpointManager(
        ckpt, model_dir, config.train.keep_checkpoint_max)

    # Setup metrics.
    self._metrics = {
        'total_num_params': tf.keras.metrics.Mean('total_num_params'),
        'grad_global_norm': tf.keras.metrics.Mean('grad_global_norm'),
        'weight_linf_norm': tf.keras.metrics.Mean('weight_linf_norm'),
        'loss': tf.keras.metrics.Mean('loss'),
    }
    self._metrics.update({
        f'loss_{t.name}': tf.keras.metrics.Mean(f'loss_{t.name}')
        for t in config.tasks})
    self._print_params = True

  def train_step(self, examples, tasks, strategy):
    """Defines a single training step for model update given examples and tasks.

    Args:
      examples: a list of data examples to be fed into the paired task class for
        preprocessing.
      tasks: a list of tasks that provide preprocessing and postprocessing for
        specific task.
      strategy: tensorflow strategy such as `TPUStrategy` or `MirroredStrategy`.
    """
    logging.info('train_step begins...')
    preprocessed_outputs = [
        t.preprocess_batched(e, training=True) for e, t in zip(examples, tasks)]

    task_loss_metrics = {}
    with tf.GradientTape() as tape:
      loss = 0
      for o, task in zip(preprocessed_outputs, tasks):
        loss_t = self.compute_loss(o)
        task_loss_metrics[f'loss_{task.config.task.name}'] = loss_t
        loss += loss_t * task.config.task.weight
      trainable_variables = self._model.trainable_variables
      grads = tape.gradient(  # div by num_replicas_in_sync for mean gradient.
          loss / strategy.num_replicas_in_sync, trainable_variables)
      self._optimizer.apply_gradients(zip(grads, trainable_variables))

    # Update metrics.
    self._metrics['loss'].update_state(loss)
    for k, v in task_loss_metrics.items():
      self._metrics[k].update_state(v)
    wmx = [tf.reduce_max(tf.math.abs(m)) for m in trainable_variables]
    self._metrics['weight_linf_norm'].update_state(tf.reduce_max(wmx))
    multiplier = strategy.num_replicas_in_sync
    self._metrics['grad_global_norm'].update_state(tf.linalg.global_norm(
        [tf.math.scalar_mul(multiplier, g) for g in grads if g is not None]))
    self._metrics['total_num_params'].update_state(
        utils.count_params(self._model, verbose=self._print_params))
    self._print_params = False
    logging.info('train_step ends...')

  @abc.abstractmethod
  def compute_loss(self, preprocessed_outputs):
    """Compute loss based on model outputs and targets."""

  def check_checkpoint_restored(self):
    """Check if the checkpoints are correctely restored."""
    (verify_restored,), (verify_restored_p,) = (
        utils.check_checkpoint_restored(
            [self._verify_restored], [self._verify_restored_p]))
    self._verify_restored = verify_restored
    self._verify_restored_p = verify_restored_p

  def reset(self):
    """Reseting the metrics and/or other state accumulators."""
    for k, _ in self._metrics.items():
      self._metrics[k].reset_states()

  @property
  def model(self):
    """Returns model instance."""
    return self._model

  @property
  def optimizer(self):
    """Returns optimizer instance."""
    return self._optimizer

  @property
  def learning_rate(self):
    """Returns learning rate scheduling instance."""
    return self._learning_rate

  @property
  def metrics(self):
    """Returns metrics instance."""
    return self._metrics

  @property
  def config(self):
    """Returns config instance."""
    return self._config

  @property
  def checkpoint_manager(self):
    """Returns checkpoint_manager instance."""
    return self._checkpoint_manager
