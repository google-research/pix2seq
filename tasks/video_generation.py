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
"""Task for video generation."""
from absl import logging
import ml_collections
import utils
from data import data_utils
from metrics import fvd
from tasks import task as task_lib
import tensorflow as tf


@task_lib.TaskRegistry.register('video_generation')
class TaskVideoGeneration(task_lib.Task):  # pytype: disable=base-class-error
  """Task for video generation."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self._metrics = {}
    self._tfgan_evaluator = fvd.FVDMetricEvaluator(
        dataset_name=config.dataset.tfds_name,
        image_size=config.dataset.image_size)

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

    def _preprocess_single_example(features, labels):
      features_list, labels_list = [], []
      for _ in range(batch_duplicates if training else 1):
        video = preprocess_video(
            features['video'],
            height=image_size,
            width=image_size,
            seq_len=self.config.dataset.get('seq_len'),
            cropping=self.config.dataset.cropping,
            flipping=self.config.dataset.flipping,
            training=training)
        label = tf.one_hot(labels['label'], self.config.dataset.num_classes)
        features_list.append({'video': video})
        labels_list.append({'label': label})
      features = utils.merge_list_of_dict(features_list)
      labels = utils.merge_list_of_dict(labels_list)
      return features, labels

    dataset = dataset.map(_preprocess_single_example,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def preprocess_batched(self, batched_examples, training):
    """Task-specific preprocessing of batched examples on accelerators (TPUs).

    Args:
      batched_examples: tuples of feature and label tensors that are
        preprocessed, batched, and stored with `dict`.
      training: bool.

    Returns:
      videos: `float` of shape (bsz, t, h, w, c)
      labels: `int` of shape (bsz)
    """
    features, labels = batched_examples
    if training:
      return features['video'], labels['label']
    else:
      return features['video'], labels['label'], batched_examples

  def infer(self, model, preprocessed_outputs, **kwargs):
    """Perform inference given the model and preprocessed outputs."""
    videos, labels, examples = preprocessed_outputs
    samples = model.sample(
        num_samples=tf.shape(videos)[0],
        iterations=self.config.model.infer_iterations,
        method=self.config.model.sampler_name,
        images=videos,
        labels=labels,
        **kwargs)
    return examples, samples

  def postprocess_tpu(self,
                      batched_examples,
                      samples,
                      training=False):
    """Organizing results after fitting the batched examples in graph.

    Such as updating metrics, putting together results for computing metrics in
      CPU/numpy mode.

    Note: current implementation only support eval mode where gt are given in
      metrics as they are not constructed here from input_seq/target_seq.

    Args:
      batched_examples: a tupple of features (`dict`) and labels (`dict`),
        containing videos and labels.
      samples: `float` predicted video tensor of (bsz, t, h, w, c).
      training: `bool` indicating training or inference mode.

    Returns:
      results for passing to `postprocess_cpu` which runs in CPU mode.
    """
    logging.info('Start postprocess_tpu.')
    features, labels = batched_examples
    videos = features['video']
    labels = tf.argmax(labels['label'], -1)

    # FID
    data_real, data_gen = self._tfgan_evaluator.preprocess_inputs(
        [videos, samples], is_n1p1=False)
    (logits_real, pool3_real), (logits_gen, pool3_gen) = (
        self._tfgan_evaluator.get_inception_stats([data_real, data_gen]))

    logging.info('postprocess_tpu done.')
    return (videos, samples, logits_real, pool3_real, logits_gen, pool3_gen)

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
      ret_results: whether to return visualization videos.

    Returns:
      A dict of visualization videos if ret_results, else None.
    """
    logging.info('Start postprocess_cpu')
    videos, samples, logits_real, pool3_real, logits_gen, pool3_gen = outputs

    # FID update.
    self._tfgan_evaluator.update_stats(
        logits_real, pool3_real, logits_gen, pool3_gen)

    # videos summary.
    bsz, t, h, w, c = utils.shape_as_list(samples)
    a = tf.cast(tf.math.sqrt(tf.cast(bsz, tf.float32)), tf.int32)
    b = a
    vis_samples = samples[:a * a, ...]
    vis_samples = tf.reshape(vis_samples, [a, b, t, h, w, c])
    vis_samples = tf.transpose(vis_samples, [0, 3, 1, 2, 4, 5])
    videos_sum = tf.reshape(vis_samples, [1, a * h, b * t * w, c])
    if eval_step < 2:
      tf.summary.image(
          f'{summary_tag}/samples_{eval_step}', videos_sum, step=train_step)

    logging.info('postprocess_cpu done.')
    if ret_results:
      return {'gt': videos, 'pred': samples}

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


def preprocess_video(video,
                     height,
                     width,
                     seq_len=None,
                     cropping='none',
                     flipping='none',
                     training=False):
  """Preprocesses the given video.

  Args:
    video: `Tensor` representing an video of arbitrary size (B x H x W x C).
    height: Height of output video.
    width: Width of output video.
    seq_len: Length of sequence to crop.
    cropping: which cropping to apply to the video.
    flipping: which flipping to apply to the video.
    training: `bool` for whether the preprocessing is for training.

  Returns:
    A preprocessed video `Tensor` of range [0, 1].
  """
  seq_len = seq_len if seq_len is not None else video.shape[0]
  video = tf.image.convert_image_dtype(video, dtype=tf.float32)
  if cropping == 'center':
    video = data_utils.largest_center_square(video)
    video = tf.image.resize(
        video,
        size=(height, width),
        method='area',
        preserve_aspect_ratio=False,
        antialias=True)
  elif cropping == 'random':
    video = data_utils.crop_video(
        frames=video,
        height=height,
        width=width,
        seq_len=seq_len,
        random=True)
  elif cropping == 'random_resize':
    video = tf.image.resize(
        video,
        size=(int(height*1.25), int(width*1.25)),   # TODO ajabri: make general
        method='area',
        preserve_aspect_ratio=False,
        antialias=True)
    video = data_utils.crop_video(
        frames=video,
        height=height,
        width=width,
        seq_len=seq_len,
        random=True)
  elif cropping != 'none':
    raise ValueError(f'Unknown cropping method {cropping}')
  if training:
    if flipping == 'left_right':
      video = tf.image.random_flip_left_right(video)
    elif flipping != 'none':
      raise ValueError(f'Unknown flipping method {flipping}')
  video = tf.reshape(video, [seq_len, height, width, 3])   # let arch know shape

  return video
