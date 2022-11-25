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
"""Tokenizer library."""

import abc

import tensorflow as tf
import tensorflow_text as tf_text


class Tokenizer(abc.ABC):
  """Tokenizer base class."""

  def __init__(self):
    pass

  @property
  @abc.abstractmethod
  def vocab_size(self):
    """Vocab size."""

  @abc.abstractmethod
  def string_to_ids(self, string):
    """Tokenize a single string."""

  @abc.abstractmethod
  def strings_to_ids(self, strings):
    """Tokenize a batch of strings."""

  @abc.abstractmethod
  def ids_to_strings(self, ids, ids_len):
    """Detokenize a batch of ids."""


class SPTokenizer(Tokenizer):
  """Sentence Piece Tokenizer."""

  def __init__(self, model_path, add_bos=False, add_eos=False):
    super(SPTokenizer, self).__init__()
    self.model_path = model_path
    with tf.io.gfile.GFile(model_path, "rb") as f:
      model = f.read()
      self.tokenizer = tf_text.SentencepieceTokenizer(model,
                                                      out_type=tf.string,
                                                      add_bos=add_bos,
                                                      add_eos=add_eos)

  @property
  def vocab_size(self):
    return int(self.tokenizer.vocab_size().numpy())

  def string_to_ids(self, string):
    tokens = self.tokenizer.tokenize(string)
    pieces = self.tokenizer.string_to_id(tokens)
    return tf.cast(pieces, tf.int64)

  def strings_to_ids(self, strings):
    return self.string_to_ids(strings)

  def ids_to_strings(self, ids, ids_len):
    return self.tokenizer.detokenize(ids)
