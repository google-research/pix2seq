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
"""Object recognition / image classification task."""

import ml_collections
import utils
from tasks import task as task_lib
from simclr.tf2 import data_util as simclr_data_util
import tensorflow as tf


@task_lib.TaskRegistry.register('object_recognition')
class TaskObjectRecognition(task_lib.Task):
  """Object recognition."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self._metrics = {
        'accuracy': tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    }
    if 'linear_eval_all_layers' in config.model and (
        config.model.linear_eval_all_layers):
      for i in range(config.model.num_encoder_layers+2):
        self._metrics['accuracy_%d'%i] = (
            tf.keras.metrics.SparseCategoricalAccuracy('accuracy_%d'%i))

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
    config = self.config.task
    image_size = self.config.dataset.image_size

    def _preprocess_single_example(examples):
      test_crop = False if image_size <= 32 else True

      examples_list = []
      for _ in range(batch_duplicates if training else 1):
        image_ = preprocess_image(
            examples['image'],
            height=image_size,
            width=image_size,
            training=training,
            color_jitter_strength=config.color_jitter_strength,
            test_crop=test_crop,
            train_crop=config.train_crop)
        if config.get('set_pixel_range_minus_one_to_one'):
          image_ = image_ * 2 - 1
        if examples['label'].shape.ndims == 0:
          label_ = tf.one_hot(examples['label'],
                              self.config.dataset.num_classes)
        else:
          label_ = examples['label']
        examples_list.append({'image': image_, 'label': label_})
      examples = utils.merge_list_of_dict(examples_list)
      return examples

    dataset = dataset.map(_preprocess_single_example,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def preprocess_batched(self, examples, training):
    """Task-specific preprocessing of batched examples on accelerators (TPUs).

    Args:
      examples: `dict` of images and labels.
      training: bool.

    Returns:
      images: `float` of shape (bsz, h, w, c)
      labels: `int` of shape (bsz)
    """
    if training:
      return examples['image'], examples['label']
    else:
      return examples['image'], examples['label'], examples

  def infer(self, model, preprocessed_outputs):
    """Perform inference given the model and preprocessed outputs."""
    image, _, examples = preprocessed_outputs  # response_seq unused by default
    outputs = model(image, training=False)
    logits = outputs[0]
    if hasattr(model, 'encode_decode'):
      outputs = model.encode_decode(image, training=False)
      images_recon = outputs[0]
    elif hasattr(model, 'sample'):
      outputs = model.sample(num_samples=tf.shape(image)[0])
      images_recon = outputs
    else:
      images_recon = image
    return examples, logits, images_recon

  def postprocess_tpu(self, examples, logits, images_recon,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                      training=False):
    """Organizing results after fitting the batched examples in graph.

    Such as updating metrics, putting together results for computing metrics in
      CPU/numpy mode.

    Note: current implementation only support eval mode where gt are given in
      metrics as they are not constructed here from input_seq/target_seq.

    Args:
      examples: `dict` containing images and labels.
      logits: `float` sequence of shape (bsz, seqlen', vocab_size).
      images_recon: `float` predicted image tensor of (bsz, h, w, c).
      training: `bool` indicating training or inference mode.

    Returns:
      results for passing to `postprocess_cpu` which runs in CPU mode.
    """
    images = examples['image']
    labels = tf.argmax(examples['label'], -1)
    if isinstance(logits, list):
      self._metrics['accuracy'].update_state(labels, logits[-1])
      if len(logits) > 1:
        assert self.config.model.linear_eval_all_layers
        for i, logits_ in enumerate(logits):
          self._metrics['accuracy_%d'%i].update_state(labels, logits_)
    else:
      self._metrics['accuracy'].update_state(labels, logits)
    return (images, images_recon)

  def postprocess_cpu(self, outputs, train_step,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                      eval_step=None, training=False, summary_tag='eval',
                      ret_results=False):
    """CPU post-processing of outputs.

    Args:
      outputs: a tuple of tensor passed from `postprocess_tpu`.
      train_step: `int` scalar indicating training step of current model or
        the checkpoint.
      eval_step: `int` scalar indicating eval step for the given checkpoint.
      training: `bool` indicating training or inference mode.
      summary_tag: `string` of name scope for result summary.
      ret_results: whether to return visualization images.

    Returns:
      A dict of visualization images if ret_results, else None.
    """
    images, images_recon = outputs
    if self.config.task.get('image_gen_sum'):
      bsz, h, w, c = utils.shape_as_list(images_recon)
      a = tf.cast(tf.math.sqrt(tf.cast(bsz, tf.float32)), tf.int32)
      b = bsz // a
      images_recon = tf.reshape(images_recon, [a, b, h, w, c])
      images_recon = tf.transpose(images_recon, [0, 2, 1, 3, 4])
      images_sum = tf.reshape(images_recon, [1, a * h, b * w, c])
    else:
      images_sum = tf.concat([images, images_recon], 2)
    if self.config.task.get('set_pixel_range_minus_one_to_one'):
      norm = lambda x: (x-tf.reduce_min(x))/(tf.reduce_max(x)-tf.reduce_min(x))
      images_sum = norm(images_sum)
    if eval_step <= 5:
      tf.summary.image(summary_tag + '/gt_pred', images_sum, step=train_step)
    if ret_results:
      return {'gt': images, 'pred': images_recon}

  def compute_scalar_metrics(self, step):
    """Returns a dict containing scalar metrics to log."""
    result = {}
    for metric in self._metrics.values():
      result[metric.name] = metric.result().numpy()
    return result

  def reset_metrics(self):
    """Reset states of metrics accumulators."""
    for metric in self._metrics.values():
      metric.reset_states()


def preprocess_image(image, height, width, training=False,
                     color_jitter_strength=0., test_crop=True, train_crop=True):
  """Preprocesses the given image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    training: `bool` for whether the preprocessing is for training.
    color_jitter_strength: `float` between 0 and 1 indicating the color
      distortion strength, disable color distortion if not bigger than 0.
    test_crop: whether or not to extract a central crop of the images
        (as for standard ImageNet evaluation) during the evaluation.
    train_crop: whether or not to apply random crop during training.

  Returns:
    A preprocessed image `Tensor` of range [0, 1].
  """
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if image.shape[-1] == 1:
    image = tf.image.grayscale_to_rgb(image)
  if training:
    return simclr_data_util.preprocess_for_train(
        image, height, width, color_jitter_strength, train_crop)
  else:
    return simclr_data_util.preprocess_for_eval(image, height, width, test_crop)
