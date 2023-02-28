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
"""Evaluation metrics for FVD/IS scores with I3D / C3D nets."""

import dataclasses
from functools import partial

from absl import logging
from jax.experimental import jax2tf
import numpy as np
from metrics.fid import get_fid_score
from metrics.fid import get_stats_for_fid
from metrics.fid import TFGANMetricEvaluator

import tensorflow as tf
import tensorflow_gan as tfgan
from universal_diffusion.metrics import c3d
from universal_diffusion.metrics import i3d


@dataclasses.dataclass
class FVDMetricEvaluator(TFGANMetricEvaluator):
  """A wrapper class for tensorflow-gan evaluation extended for FVD."""
  dataset_name: str
  image_size: int = -1
  activations_key: str = 'pool_3'

  def __post_init__(self):
    self.all_logits_real = []
    self.all_pool3_real = []
    self.all_logits_gen = []
    self.all_pool3_gen = []
    self.dataset_stats_mean, self.dataset_stats_cov = self.load_fid_stats()

    if self.dataset_name == 'ucf101':
      self.model = jax2tf.convert(partial(c3d.run_model, c3d.load_params()))
    elif self.dataset_name == 'kinetics600':
      self.model = jax2tf.convert(partial(i3d.run_model, i3d.load_params()))
    else:
      assert False, 'dataset not supported for FVD'

  def load_fid_stats(self, stat_path=None):
    """Load the pre-computed dataset statistics."""
    logging.info('loading FID stats for datasets %s', self.dataset_name)
    if self.dataset_name in {'ucf101', 'kinetics600'} and False:
      assert self.image_size in [64, 128]
      filename = '{}/{}_{}_stats_real.npz'.format(
          stat_path, self.dataset_name, self.image_size)
      with tf.io.gfile.GFile(filename, 'rb') as fin:
        stats_real = np.load(fin)
        logging.info('FID stats loading done! Number of examples %d',
                     stats_real['mu'].shape[0])
        return stats_real['mu'], stats_real['cov']
    else:
      logging.warn('Dataset %s stats not found!', self.dataset_name)
      return None, None

  def preprocess_inputs(self, inputs, is_n1p1=False):
    if isinstance(inputs, list):
      all_inputs = tf.concat(inputs, 0)
      all_inputs = self.preprocess_inputs(all_inputs, is_n1p1=is_n1p1)
      return tf.split(all_inputs, len(inputs))
    if is_n1p1:
      inputs = tf.clip_by_value(inputs, -1.0, 1.0)
      inputs = (inputs + 1.0) / 2.0
    inputs = inputs * 255.0
    return inputs

  def get_inception_stats(self, inputs):
    if isinstance(inputs, list):
      return [self.get_inception_stats(x) for x in inputs]
    stats = self.model(inputs)
    if 'features' in stats:
      return stats['logits'], stats['features']
    else:
      return stats['logits_mean'], stats['pool']

  def compute_fid_score(self):
    """Return a dict of metrics."""
    metrics = {}
    logging.info('Computing Inception score.')
    all_logits_gen = np.concatenate(self.all_logits_gen, axis=0)
    all_logits_real = np.concatenate(self.all_logits_real, axis=0)
    logging.info('IS number of gen samples: %d, number of classes: %d',
                 all_logits_gen.shape[0], all_logits_gen.shape[1])
    is_score = tfgan.eval.classifier_score_from_logits(all_logits_gen)
    metrics.update({'inception_score': is_score})

    logging.info('Computing FVD score.')
    all_stats_real = np.concatenate(self.all_pool3_real, axis=0)
    all_stats_gen = np.concatenate(self.all_pool3_gen, axis=0)
    logging.info('FVD number of real samples: %d', all_stats_real.shape[0])
    logging.info('FVD number of generated samples: %d', all_stats_gen.shape[0])
    gen_mean, gen_cov = get_stats_for_fid(all_stats_gen)
    ref_mean, ref_cov = get_stats_for_fid(all_stats_real)

    gen_logits_mean, gen_logits_cov = get_stats_for_fid(all_logits_gen)
    ref_logits_mean, ref_logits_cov = get_stats_for_fid(all_logits_real)

    metrics.update({
        'fvd_pool_batch': get_fid_score(gen_mean, gen_cov, ref_mean, ref_cov),
        'fvd_batch': get_fid_score(gen_logits_mean, gen_logits_cov,
                                   ref_logits_mean, ref_logits_cov),
    })
    if self.dataset_stats_mean is not None:
      metrics.update({
          'fvd_pool_full':
              get_fid_score(gen_mean, gen_cov, self.dataset_stats_mean,
                            self.dataset_stats_cov),
          'fvd_pool_batch_vs_full':
              get_fid_score(ref_mean, ref_cov, self.dataset_stats_mean,
                            self.dataset_stats_cov),
      })
    return metrics
