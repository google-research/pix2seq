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
"""Task base class."""

import abc
from absl import logging
import ml_collections
import registry
import tensorflow as tf

TaskRegistry = registry.Registry()


class Task(abc.ABC):
  """Task class.

  Providing:
    - Preprocessing functions for a specific task that turns raw features into
      inputs in a common interface.
    - Post-processing functions for a specific task that decode the model's
      outputs in a common interface.
    - Evaluation for a specific task.
    - Important task properties, such as vocab size, max seq len.
  """

  def __init__(self,
               config: ml_collections.ConfigDict):
    self.config = config

  @property
  def task_vocab_id(self):
    return self.config.task.vocab_id

  @abc.abstractmethod
  def preprocess_single(self, dataset: tf.data.Dataset, batch_duplicates: int,
                        training: bool):
    """Task-specific preprocessing of individual example in the dataset.

    Args:
      dataset: A tf.data.Dataset.
      batch_duplicates: `int`, enlarge a batch by augmenting it multiple times
        (as specified) and concating the augmented examples.
      training: bool.

    Returns:
      A dataset.
    """

  @abc.abstractmethod
  def preprocess_batched(self, batched_examples, training):
    """Task-specific preprocessing of batched examples on accelerators (TPUs).

    Args:
      batched_examples: preprocessed and batched examples.
      training: bool.

    Returns batched inputs in a comon interface for modeling.
    """

  @abc.abstractmethod
  def postprocess_tpu(self):
    """Task-specific post processing on accelerators (TPUs).

    This is intended for inference / evaluation time only.

    Returns a list of tensors for `postprocess_cpu` to further process.
    """

  @abc.abstractmethod
  def postprocess_cpu(self):
    """Task-specific post processing on CPUs.

    This is intended for inference / evaluation time only.

    It receives outputs from `postprocess_tpu`, further processes, and update
      internal states (e.g. _metrics).
    """

  def _log_metrics(self, metrics_dict, step):
    for key, value in metrics_dict.items():
      logging.info('Step: [%d] %s = %f', step, key, value)
      tf.summary.scalar(key, value, step)

  def evaluate(self, summary_writer, step, eval_tag):
    """Evaluate results on accumulated outputs (after multiple infer steps).

    Args:
      summary_writer: the summary writer.
      step: current step.
      eval_tag: `string` name scope for eval result summary.

    Returns:
      result as a `dict`.
    """
    metrics = self.compute_scalar_metrics(step)
    with summary_writer.as_default():
      with tf.name_scope(eval_tag):
        self._log_metrics(metrics, step)
      summary_writer.flush()
    self.reset_metrics()
    return metrics

  @abc.abstractmethod
  def compute_scalar_metrics(self, step):
    """Returns a dict containing scalar metrics to log."""

  @abc.abstractmethod
  def reset_metrics(self):
    """Reset states of metrics accumulators."""
