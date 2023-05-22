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
"""Instance segmentation task via COCO metric evaluation."""

import ml_collections

import utils
import vocab
from data import tokenizer as tokenizer_lib
from metrics import metric_registry
from tasks import task as task_lib
from tasks import task_utils
import tensorflow as tf


@task_lib.TaskRegistry.register('captioning')
class TaskCaptioning(task_lib.Task):
  """Image captioning with coco evaluation."""

  def __init__(self,
               config: ml_collections.ConfigDict):
    super().__init__(config)
    metric_config = config.task.get('metric')
    if metric_config and metric_config.get('name'):
      self._coco_metrics = metric_registry.MetricRegistry.lookup(
          metric_config.name)(config)
    else:
      self._coco_metrics = None
    self._tokenizer = tokenizer_lib.SPTokenizer(
        config.tokenizer.sentencepiece_model,
        add_bos=config.tokenizer.add_bos,
        add_eos=config.tokenizer.add_eos)

  def preprocess_single(self, dataset, batch_duplicates, training):
    """Task-specific preprocessing of individual example in the dataset.

    Args:
      dataset: A tf.data.Dataset.
      batch_duplicates: `int`, enlarge a batch by augmenting it multiple times
        (as specified) and concating the augmented examples.
      training: bool.

    Returns:
      A dataset.
    """
    if batch_duplicates > 1:
      raise NotImplementedError('Not supporting batch_duplicate=%d > 1 for '
                                'caption as of now.' % batch_duplicates)

    def _preprocess_single_example(example):
      config = self.config.task
      mconfig = self.config.model
      if training:
        captions = []
        for i in range(config.captions_per_image):
          caption = (self._tokenizer.string_to_ids(example['captions'][i]) +
                     mconfig.text_vocab_shift)
          captions.append(utils.pad_to_max_len(caption, config.max_seq_len, -1))
        captions = tf.stack(captions)

        for t in self.train_transforms:
          example = t.process_example(example)
        example['captions'] = captions
      else:
        for t in self.eval_transforms:
          example = t.process_example(example)

        # Use the first caption. This  won't be used in eval.
        caption = (self._tokenizer.string_to_ids(example['captions'][0]) +
                   mconfig.text_vocab_shift)
        caption = utils.pad_to_max_len(caption, config.max_seq_len, -1)
        example['captions'] = caption
      return example

    dataset = dataset.map(_preprocess_single_example,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def preprocess_batched(self, batched_examples, training):
    """Batched preprocessing & sequence construction.

    Args:
      batched_examples: a tupple of features (`dict`) and labels (`dict`),
        containing images and labels.
      training: `bool` indicating training or inference mode.

    Returns:
      images: `float` of shape (bsz, h, w, c)
      input_seq: `int` of shape (bsz, seqlen), or (bsz, instacnes, seqlen)
        for multiple instances as in keypoint for example.
      target_seq: `int` of shape (bsz, seqlen'), or (bsz, instacnes, seqlen')
        for multiple instances as in keypoint for example.
    """
    config = self.config.task
    mconfig = self.config.model

    if training:
      response_seq = batched_examples['captions']  # (bsz, num_cap, max_seq_len)
      prompt_seq = task_utils.build_prompt_seq_from_task_id(
          self.task_vocab_id, response_seq)  # (bsz, 1)
      label_seq = tf.concat([prompt_seq, response_seq], -1)
      token_weights = tf.where(
          response_seq == 1 + mconfig.text_vocab_shift,  # eos token
          config.eos_token_weight, 1.0)
      input_seq, target_seq = label_seq[..., :-1], label_seq[..., 1:]

      if config.input_seq_drop_rate > 0:
        input_seq = tf.where(
            tf.random.uniform(tf.shape(input_seq)) > config.input_seq_drop_rate,
            input_seq, vocab.FAKE_TEXT_TOKEN)

      return batched_examples['image'], input_seq, target_seq, token_weights
    else:
      return (batched_examples['image'], batched_examples['captions'],
              batched_examples)

  def infer(self, model, preprocessed_outputs):
    """Perform inference given the model and preprocessed outputs."""
    config = self.config.task
    image, _, examples = preprocessed_outputs  # response_seq unused by default
    bsz = tf.shape(image)[0]
    prompt_seq = task_utils.build_prompt_seq_from_task_id(
        self.task_vocab_id, prompt_shape=(bsz, 1))
    pred_seq, logits, _ = model.infer(
        image, prompt_seq, encoded=None,
        temperature=config.temperature, top_k=config.top_k, top_p=config.top_p)
    # if True:  # Sanity check by using gt response_seq as pred_seq.
    #   pred_seq = response_seq
    #   logits = tf.one_hot(pred_seq, self.vocab_size)
    return examples, pred_seq, logits

  def postprocess_tpu(self, batched_examples, pred_seq, logits, training=False):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Organizing results after fitting the batched examples in graph.

    Such as updating metrics, putting together results for computing metrics in
      CPU/numpy mode.

    Args:
      batched_examples: a tupple of features (`dict`) and labels (`dict`),
        containing images and labels.
      pred_seq: `int` sequence of shape (bsz * instances, seqlen').
      logits: `float` sequence of shape (bsz * instances, seqlen', vocab_size).
      training: `bool` indicating training or inference mode.

    Returns:
      results for passing to `postprocess_cpu` which runs in CPU mode.
    """
    return (batched_examples['image'], batched_examples['image/id'],
            batched_examples['captions'], pred_seq)

  def postprocess_cpu(self, outputs, train_step,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                      eval_step=None, training=False, summary_tag='eval',
                      ret_results=False):
    """CPU post-processing of outputs.

    Such as computing the metrics, log image summary.

    Note: current implementation only support eval mode where gt are given in
      metrics as they are not given here in outputs.

    Args:
      outputs: a tuple of tensor passed from `postprocess_tpu`.
      train_step: `int` scalar indicating training step of current model or
        the checkpoint.
      eval_step: `int` scalar indicating eval step for the given checkpoint.
      training: `bool` indicating training or inference mode.
      summary_tag: `string` of name scope for result summary.
      ret_results: whether to return visualization images/captions.

    Returns:
      A dict of visualization images/caption if ret_results, else None.
    """
    config = self.config.task
    mconfig = self.config.model
    del summary_tag
    if not training:
      images, image_ids, gt_seq, pred_seq = outputs
      batch_size = tf.shape(image_ids)[0]
      pred_seq = tf.where(pred_seq == 0, 0,
                          pred_seq - mconfig.text_vocab_shift)
      original_pred_seq = pred_seq.numpy()
      pred_seq = tf.minimum(
          tf.maximum(pred_seq, 0), self._tokenizer.vocab_size - 1)
      pred_seq = tf.cast(pred_seq, tf.int32)
      clipped_pred_seq = pred_seq.numpy()
      output_text = self._tokenizer.ids_to_strings(
          pred_seq,
          tf.ones([batch_size], tf.int32) * config.max_seq_len).numpy()
      output_text = [o.decode('utf-8') for o in output_text]
      if self._coco_metrics:
        self._coco_metrics.record_prediction(image_ids.numpy(), output_text,
                                             original_pred_seq,
                                             clipped_pred_seq)

      if ret_results:
        gt_seq = tf.where(gt_seq == 0, 0,
                          gt_seq - mconfig.text_vocab_shift)
        gt_text = self._tokenizer.ids_to_strings(
            tf.cast(gt_seq, tf.int32),
            tf.ones([batch_size], tf.int32) * config.max_seq_len).numpy()
        gt_text = [s.decode('utf-8') for s in gt_text]
        return {'gt_images': images,
                'gt_captions': gt_text,
                'pred_captions': output_text}

  def compute_scalar_metrics(self, step):
    """Returns a dict containing scalar metrics to log."""
    if self._coco_metrics:
      return self._coco_metrics.result(step)
    else:
      return {}

  def reset_metrics(self):
    """Reset states of metrics accumulators."""
    if self._coco_metrics:
      self._coco_metrics.reset_states()
