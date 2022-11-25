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
"""Dataset base class."""

import abc
import functools
import operator
from typing import Callable
import ml_collections

import registry
import tensorflow as tf
import tensorflow_datasets as tfds

DatasetRegistry = registry.Registry()


class Dataset(abc.ABC):
  """A dataset that handles creating a tf.data.Dataset."""

  def __init__(self, config: ml_collections.ConfigDict):
    """Constructs the dataset."""
    self.config = config.dataset
    self.task_config = config.task

  @abc.abstractmethod
  def extract(self, example, training):
    """Extracts needed features & annotations into a flat dictionary.

    Note: be consisous about 0 in label, which should probably reserved for
       special use (such as padding).

    Args:
      example: `dict` of raw features.
      training: `bool` of training vs eval mode.

    Returns:
      example: `dict` of relevant features and labels
    """

  @abc.abstractmethod
  def load_dataset(self, input_context, training):
    """Load tf.data.Dataset from sources such as TFDS or TFRecord files."""

  def parse_example(self, example):
    return example

  def filter_example(self, unused_example, unused_training):
    return True

  def pipeline(self,
               process_single_example: Callable[[tf.data.Dataset, int, bool],
                                                tf.data.Dataset],
               global_batch_size: int, training: bool):
    """Data pipeline from name to preprocessed examples.

    Args:
      process_single_example: a function that takes single example dataset and
        returns processed example dataset.
      global_batch_size: global batch size.
      training: training vs eval mode.

    Returns:
      tf.data.Dataset instance.
    """
    config = self.config
    def input_fn(input_context):
      dataset = self.load_dataset(input_context, training)
      if config.cache_dataset:
        dataset = dataset.cache()

      if input_context:
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        # Sharding is not neccesary for TFDS given read_config above.
        # dataset = dataset.shard(input_context.num_input_pipelines,
        #                         input_context.input_pipeline_id)
      else:
        batch_size = global_batch_size

      if training:
        options = tf.data.Options()
        options.deterministic = False
        options.experimental_slack = True
        dataset = dataset.with_options(options)
        buffer_size = config.get('buffer_size', 0)
        if buffer_size <= 0:
          buffer_size = 10 * batch_size
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.repeat()

      dataset = dataset.map(
          self.parse_example,
          num_parallel_calls=tf.data.experimental.AUTOTUNE
      ).filter(
          lambda x: self.filter_example(x, training)
      ).map(
          lambda x: self.extract(x, training),
          num_parallel_calls=tf.data.experimental.AUTOTUNE
      )
      if process_single_example:
        dataset = process_single_example(
            dataset, config.batch_duplicates, training)

      # TODO(b/181662974): Revert this and support non-even batch sizes.
      # dataset = dataset.batch(batch_size, drop_remainder=training)
      dataset = dataset.padded_batch(batch_size, drop_remainder=True)
      if config.batch_duplicates > 1 and training:
        dataset = dataset.map(self._flatten_dims,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
      return dataset

    return tf.distribute.get_strategy().distribute_datasets_from_function(
        input_fn)

  def _flatten_dims(self, features, labels):
    """Flatten first 2 dims when batch is independently duplicated."""

    def flatten_first_2_dims(t):
      """Merge first 2 dims."""
      shape_list = t.shape.as_list()
      new_bsz = functools.reduce(operator.mul, shape_list[:2])
      out_shape = [new_bsz] + shape_list[2:]
      return tf.reshape(t, out_shape)

    features = {k: flatten_first_2_dims(v) for k, v in features.items()}
    labels = {k: flatten_first_2_dims(v) for k, v in labels.items()}
    return features, labels

  @property
  @abc.abstractmethod
  def num_train_examples(self):
    """Number of training examples."""

  @property
  @abc.abstractmethod
  def num_eval_examples(self):
    """Number of eval examples."""


class TFDSDataset(Dataset):
  """A dataset created from a TFDS dataset.

    Each example is a dictionary, but the fields may be different for each
    dataset.

    Each task would have a list of required fields (e.g. bounding boxes for
    object detection). When a dataset is used for a specific task, it should
    contain all the fields required by that task.
  """

  def __init__(self, config: ml_collections.ConfigDict):
    """Constructs the dataset."""
    super().__init__(config)
    self.builder = tfds.builder(self.config.tfds_name,
                                data_dir=self.config.get('data_dir', None))
    self.builder.download_and_prepare()
    self.allowed_tasks = []

  def load_dataset(self, input_context, training):
    """Load tf.data.Dataset from TFDS."""
    split = self.config.train_split if training else self.config.eval_split
    # For TFDS, pass input_context using read_config to make TFDS read
    # different parts of the dataset on different workers.
    read_config = tfds.ReadConfig(input_context=input_context)
    if isinstance(split, list):
      dataset = self.builder.as_dataset(
          split=split[0], shuffle_files=True, read_config=read_config)
      for i in range(1, len(split)):
        dataset.concatenate(self.builder.as_dataset(
            split=split[i], shuffle_files=True, read_config=read_config))
    else:
      dataset = self.builder.as_dataset(
          split=split, shuffle_files=True, read_config=read_config)
    return dataset

  @property
  def num_train_examples(self):
    return self.builder.info.splits[self.config.train_split].num_examples

  @property
  def num_eval_examples(self):
    return self.builder.info.splits[
        self.config.eval_split].num_examples if not self.task_config.get(
            'unbatch', False) else None


class TFRecordDataset(Dataset):
  """A dataset created from tfrecord files."""

  def load_dataset(self, input_context, training):
    """Load tf.data.Dataset from TFRecord files."""
    if training or self.config.eval_split == 'train':
      file_pattern = self.config.train_file_pattern
    else:
      file_pattern = self.config.val_file_pattern
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=training)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset, cycle_length=32, deterministic=not training)
    return dataset

  @abc.abstractmethod
  def get_feature_map(self):
    """Returns feature map(s) for parsing the TFExample.

    Returns a single feature map (a dict) to parse a TFEXample.
    Returns a tuple of (context feature map, sequence feature map) to parse a
    TFSequenceExample. Context features are non-sequence features, i.e.
    independent of time/frame. Sequence features have time/frame dimension.
    """

  def parse_example(self, example):
    """Parse the serialized example into a dictionary of tensors.

    Args:
      example: the serialized tf.train.Example or tf.train.SequenceExample.

    Returns:
      a dictionary of feature name to tensors.
    """
    feature_map = self.get_feature_map()
    if isinstance(feature_map, dict):
      example = tf.io.parse_single_example(example, feature_map)
    else:
      context_features, sequence_features = feature_map
      example, sequence = tf.io.parse_single_sequence_example(
          example, context_features, sequence_features)
      example.update(sequence)

    for k in example:
      if isinstance(example[k], tf.SparseTensor):
        if example[k].dtype == tf.string:
          example[k] = tf.sparse.to_dense(example[k], default_value='')
        else:
          example[k] = tf.sparse.to_dense(example[k], default_value=0)
    return example

  @property
  def num_train_examples(self):
    return self.config.train_num_examples

  @property
  def num_eval_examples(self):
    return self.config.eval_num_examples if not self.task_config.get(
        'unbatch', False) else None
