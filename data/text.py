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
"""Text datasets."""

from data import dataset as dataset_lib


@dataset_lib.DatasetRegistry.register('text')
class TextDataset(dataset_lib.TFDSDataset):
  """Dataset."""

  def extract(self, example, training):
    """Extracts needed features & annotations into a flat dictionary.

    Args:
      example: `dict` of raw features.
      training: `bool` of training vs eval mode.

    Returns:
      a sequence
    """
    if self.config.tfds_name.startswith('wikipedia'):
      text = example['title'] + '\n\n' + example['text']
    elif self.config.tfds_name.startswith('wmt'):
      src, dst = self.config.tfds_name.split('/')[1].split('-')
      if training:
        text = '[src] ' + example[src] + ' [dst] ' + example[dst]
      else:
        text = '[src] ' + example[src] + ' [dst] '
    else:
      text = example['text']
    return {'text': text}
