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
"""Metrics for text tasks."""

from metrics import metric_registry
import sacrebleu


@metric_registry.MetricRegistry.register('text_sacrebleu')
class BleuMetric():
  """BLEU metric for text."""

  def __init__(self, config):
    self._config = config
    self.reset_states()

  def reset_states(self):
    self._metric_values = None
    self._targets = []
    self._predictions = []

  def record_prediction(self, predictions, targets):
    """Records predictions.

    If multiple references are present, then each example need to have the same
    number of references.

    Args:
      predictions: list of strings. Has len batch_size.
      targets: list of strings, or list of list of strings if multiple
        references are present. Has len batch_size. In the format of
        [ex1_ref, ex2_ref, ...] or
        [[ex1_ref1, ex2_ref1, ...], [ex1_ref2, ex2_ref2, ...], ...].
    """
    self._predictions.extend(predictions)

    # Turn targets into lists.
    if not isinstance(targets[0], list):
      targets = [targets]
    if self._targets:
      assert len(self._targets) == len(targets)
      for i in range(len(targets)):
        self._targets[i].extend(targets[i])
    else:
      self._targets = targets

  def _evaluate(self):
    """Evaluates with predictions for all examples.

    Call this function from `self.result`.

    Returns:
      dict from metric name to float value.
    """
    tokenizer = self._config.get('tokenizer', 'intl')
    bleu_score = sacrebleu.corpus_bleu(self._predictions, self._targets,
                                       smooth_method='exp',
                                       smooth_value=0.0,
                                       force=False,
                                       lowercase=False,
                                       tokenize=tokenizer,
                                       use_effective_order=False)
    return {'bleu': bleu_score.score}

  def result(self):
    """Return the metric values (and compute it if needed)."""
    if self._metric_values is None:
      self._metric_values = self._evaluate()
    return self._metric_values
