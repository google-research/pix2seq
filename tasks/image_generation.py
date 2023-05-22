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
"""Task for image generation."""

import os

from absl import logging
import ml_collections
import utils
from data import data_utils
from metrics.fid import TFGANMetricEvaluator
from tasks import task as task_lib
import tensorflow as tf


@task_lib.TaskRegistry.register('image_generation')
class TaskImageGeneration(task_lib.Task):  # pytype: disable=base-class-error
  """TaskImageGeneration."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self._metrics = {}
    self._metrics['eval_loss'] = tf.keras.metrics.Mean('eval_loss')
    self._tfgan_evaluator = TFGANMetricEvaluator(
        dataset_name=config.dataset.tfds_name,
        image_size=config.dataset.image_size)

    self._write_images_to_file = config.eval.get('write_images_to_file', False)
    if self._write_images_to_file:
      self._tfrecord_dir = os.path.join(config.model_dir, 'images',
                                        config.eval.tag)
      if not tf.io.gfile.exists(self._tfrecord_dir):
        tf.io.gfile.makedirs(self._tfrecord_dir)
      self._tfrecord_writer = None

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
    image_size = self.config.dataset.image_size

    def _preprocess_single_example(examples):
      examples_list = []
      for _ in range(batch_duplicates if training else 1):
        image_ = preprocess_image(
            examples['image'],
            height=image_size,
            width=image_size,
            cropping=self.config.dataset.cropping,
            flipping=self.config.dataset.flipping,
            training=training)
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
    images, labels, examples = preprocessed_outputs
    outputs = model.sample(
        num_samples=tf.shape(images)[0],
        iterations=self.config.model.infer_iterations,
        method=self.config.model.sampler_name,
        images=images,
        labels=labels)
    if isinstance(outputs, tuple) or isinstance(outputs, list):
      samples, eval_loss = outputs
      self._metrics['eval_loss'].update_state(eval_loss)
    else:
      samples = outputs
    return examples, samples

  def postprocess_tpu(self,
                      examples,
                      samples,
                      training=False):
    """Organizing results after fitting the batched examples in graph.

    Such as updating metrics, putting together results for computing metrics in
      CPU/numpy mode.

    Note: current implementation only support eval mode where gt are given in
      metrics as they are not constructed here from input_seq/target_seq.

    Args:
      examples: `dict` of images and labels.
      samples: `float` predicted image tensor of (bsz, h, w, c).
      training: `bool` indicating training or inference mode.

    Returns:
      results for passing to `postprocess_cpu` which runs in CPU mode.
    """
    logging.info('Start postprocess_tpu.')
    # Get ground-truth images.
    if 'original_image' in examples:
      images = examples['original_image']
    elif examples['image'].shape[-1] == 3:
      images = examples['image']
    else:  # ground-truth images not available in the dataset.
      images = samples

    # FID
    data_real, data_gen = self._tfgan_evaluator.preprocess_inputs(
        [images, samples], is_n1p1=False)
    (logits_real, pool3_real), (logits_gen, pool3_gen) = (
        self._tfgan_evaluator.get_inception_stats([data_real, data_gen]))

    logging.info('postprocess_tpu done.')
    return (images, samples, logits_real, pool3_real, logits_gen, pool3_gen)

  def postprocess_cpu(self,
                      outputs,
                      train_step,
                      eval_step=None,
                      training=False,
                      summary_tag='eval',
                      ret_results=False):
    """CPU post-processing of outputs.

    Args:
      outputs: a tuple of tensor passed from `postprocess_tpu`.
      train_step: `int` scalar indicating training step of current model or the
        checkpoint.
      eval_step: `int` scalar indicating eval step for the given checkpoint.
      training: `bool` indicating training or inference mode.
      summary_tag: `string` of name scope for result summary.
      ret_results: whether to return visualization images.

    Returns:
      A dict of visualization images if ret_results, else None.
    """
    logging.info('Start postprocess_cpu')
    images, samples, logits_real, pool3_real, logits_gen, pool3_gen = outputs

    # FID update.
    self._tfgan_evaluator.update_stats(
        logits_real, pool3_real, logits_gen, pool3_gen)

    # Image summary.
    bsz, h, w, c = utils.shape_as_list(samples)
    a = tf.cast(tf.math.sqrt(tf.cast(bsz, tf.float32)), tf.int32)
    b = a
    vis_samples = samples[:a * a, ...]
    vis_samples = tf.reshape(vis_samples, [a, b, h, w, c])
    vis_samples = tf.transpose(vis_samples, [0, 2, 1, 3, 4])
    images_sum = tf.reshape(vis_samples, [1, a * h, b * w, c])
    if eval_step < 2:
      tf.summary.image(
          f'{summary_tag}/samples_{eval_step}', images_sum, step=train_step)

    # Write gt and samples to tfrecord.
    if self._write_images_to_file:
      if self._tfrecord_writer is None:
        self._tfrecord_writer = tf.io.TFRecordWriter(
            os.path.join(self._tfrecord_dir, f'step-{train_step}.tfrecord'))
      for i in range(bsz):
        ex_str = create_tf_example(images[i], samples[i])
        self._tfrecord_writer.write(ex_str)

    logging.info('postprocess_cpu done.')
    if ret_results:
      return {'gt': images, 'pred': samples}

  def compute_scalar_metrics(self, step):
    """Returns a dict containing scalar metrics to log."""
    result = {}
    for metric in self._metrics.values():
      result[metric.name] = metric.result().numpy()

    # FID
    result.update(self._tfgan_evaluator.compute_fid_score())
    return result

  def reset_metrics(self):
    """Reset states of metrics accumulators."""
    for metric in self._metrics.values():
      metric.reset_states()

    self._tfgan_evaluator.reset()
    if self._write_images_to_file:
      self._tfrecord_writer.close()
      self._tfrecord_writer = None




def create_tf_example(ref, hyp):
  """Creates a serialized tf example that stores the images.

  Args:
    ref: Tensor of [h, w, c], the ground-truth image.
    hyp: Tensor of [h, w, c], the generated image.

  Returns:
    the serialized tf example.
  """
  ref_bytes = tf.io.encode_png(
      tf.image.convert_image_dtype(ref, tf.uint8)).numpy()
  hyp_bytes = tf.io.encode_png(
      tf.image.convert_image_dtype(hyp, tf.uint8)).numpy()
  feature = {
      'hyp': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hyp_bytes])),
      'ref': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ref_bytes])),
  }
  return tf.train.Example(
      features=tf.train.Features(feature=feature)).SerializeToString()


def preprocess_image(image,
                     height,
                     width,
                     cropping='none',
                     flipping='none',
                     training=False):
  """Preprocesses the given image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    cropping: which cropping to apply to the image.
    flipping: which flipping to apply to the image.
    training: `bool` for whether the preprocessing is for training.

  Returns:
    A preprocessed image `Tensor` of range [0, 1].
  """
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if cropping == 'center':
    image = data_utils.largest_center_square_crop(image)
    image = tf.image.resize(
        image,
        size=(height, width),
        method='bicubic',
        preserve_aspect_ratio=False,
        antialias=True)
  elif cropping != 'none':
    raise ValueError(f'Unknown cropping method {cropping}')
  if training:
    if flipping == 'left_right':
      image = tf.image.random_flip_left_right(image)
    elif flipping != 'none':
      raise ValueError(f'Unknown flipping method {flipping}')
  image = tf.ensure_shape(image, [height, width, 3])   # Let arch knows shape.
  return image
