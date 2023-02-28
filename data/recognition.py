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
"""Image classification datasets."""

from data import dataset as dataset_lib
import tensorflow as tf


@dataset_lib.DatasetRegistry.register('object_recognition')
class ImageDataset(dataset_lib.TFDSDataset):
  """Dataset for image classification datasets."""

  def extract(self, example, training):
    """Extracts needed features & annotations into a flat dictionary.

    Args:
      example: `dict` of raw features.
      training: `bool` of training vs eval mode.

    Returns:
      example: `dict` of relevant features and labels.
    """
    image = example['image']
    if image.shape.rank == 2 or image.shape[-1] == 1:
      image = tf.image.grayscale_to_rgb(image)
    if 'label' in example:
      label = example['label']
    else:
      label = tf.zeros([], dtype=tf.int32)
    return {'image': image,
            'label': label}

  @property
  def num_classes(self):
    return self.builder.info.features['label'].num_classes


