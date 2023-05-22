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
"""Evaluation metrics for FID score."""

import dataclasses

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan


def get_stats_for_fid(act):
  """Get mean and std statistics from activations for FID computation."""
  if act.ndim != 2:
    raise ValueError("Expected input to have 2 axes")
  act = np.asarray(act, dtype=np.float64)
  mu = np.mean(act, axis=0)
  sigma = np.cov(act, rowvar=False)
  return mu, sigma


def _symmetric_matrix_square_root(mat, eps=1e-10):
  """Compute square root of a symmetric matrix."""
  u, s, vt = np.linalg.svd(mat, hermitian=True)
  si = np.where(s < eps, s, np.sqrt(s))
  return u.dot(np.diag(si)).dot(vt)


def _trace_sqrt_product(sigma, sigma_v):
  """Find the trace of the positive sqrt of product of covariance matrices."""
  sqrt_sigma = _symmetric_matrix_square_root(sigma)
  sqrt_a_sigmav_a = sqrt_sigma.dot(sigma_v).dot(sqrt_sigma)
  return _symmetric_matrix_square_root(sqrt_a_sigmav_a).trace()


def get_fid_score(mu1, sigma1, mu2, sigma2):
  """FID score."""
  if mu1.shape != mu2.shape:
    raise ValueError("means should have the same shape")
  dim, = mu1.shape
  if not sigma1.shape == sigma2.shape == (dim, dim):
    raise ValueError("covariance matrices should be the same shape (d, d)")
  mu1 = np.asarray(mu1, dtype=np.float64)
  mu2 = np.asarray(mu2, dtype=np.float64)
  sigma1 = np.asarray(sigma1, dtype=np.float64)
  sigma2 = np.asarray(sigma2, dtype=np.float64)
  return (np.square(mu1 - mu2).sum() + sigma1.trace() + sigma2.trace() -
          2 * _trace_sqrt_product(sigma1, sigma2))


@dataclasses.dataclass
class TFGANMetricEvaluator:
  """A wrappner class for tensorflow-gan evaluation."""
  dataset_name: str = "cifar10"
  image_size: int = -1
  inceptionv3_input_size: int = 299
  activations_key: str = "pool_3"
  resize_method: str = "bilinear"
  antialias: bool = False

  def __post_init__(self):
    self.all_logits_real = []
    self.all_pool3_real = []
    self.all_logits_gen = []
    self.all_pool3_gen = []
    self.dataset_stats_mean, self.dataset_stats_cov = self.load_fid_stats()

  def load_fid_stats(self, stats_path=None):
    """Load the pre-computed dataset statistics."""
    logging.info("loading FID stats for datasets %s", self.dataset_name)
    # TODO(iamtingchen): provide stat path via config dict.
    if self.dataset_name == "cifar10":
      filename = "{}/cifar10_stats_real.npy".format(stats_path)
    elif self.dataset_name == "downsampled_imagenet/64x64":
      filename = "{}/imagenet64_stats_real.npz".format(stats_path)
      with tf.io.gfile.GFile(filename, "rb") as fin:
        stats_real = np.load(fin)
        return stats_real["mu"], stats_real["sigma"]
    elif self.dataset_name == "imagenet2012":
      assert self.image_size in [
          32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048]
      filename = "{}/imagenet_man_{}_stats_real.npz".format(
          stats_path, self.image_size)
      with tf.io.gfile.GFile(filename, "rb") as fin:
        stats_real = np.load(fin)
        return stats_real["mu"], stats_real["cov"]
    elif self.dataset_name == "coco":
      filename = "{}/coco_stats_real.npz".format(stats_path)
      with tf.io.gfile.GFile(filename, "rb") as fin:
        stats_real = np.load(fin)
      return stats_real["mu"], stats_real["cov"]
    else:
      logging.warn("Dataset %s stats not found!", self.dataset_name)
      return None, None

    with tf.io.gfile.GFile(filename, "rb") as fin:
      stats_real = np.load(fin)
      logging.info("FID stats loading done! Number of examples %d",
                   stats_real.shape[0])
      return get_stats_for_fid(stats_real)

  def preprocess_inputs(self, inputs, is_n1p1=False):
    """Resize images and shift/clip pixels to [-1, 1]."""
    if isinstance(inputs, list):
      all_inputs = tf.concat(inputs, 0)
      all_inputs = self.preprocess_inputs(all_inputs, is_n1p1=is_n1p1)
      return tf.split(all_inputs, len(inputs))
    if is_n1p1:
      inputs = tf.clip_by_value(inputs, -1.0, 1.0)
      inputs = (inputs + 1.0) / 2.0

    inputs = tf.image.resize(
        inputs, [self.inceptionv3_input_size, self.inceptionv3_input_size],
        self.resize_method,
        antialias=self.antialias)
    inputs = tf.clip_by_value(inputs, 0.0, 1.0)
    # transform inputs to [-1, 1]
    inputs = inputs * 2 - 1.0
    return inputs

  def get_inception_stats(self, inputs):
    if isinstance(inputs, list):
      return [self.get_inception_stats(x) for x in inputs]
    stats = tfgan.eval.run_inception(inputs)
    return stats["logits"], stats["pool_3"]

  def update_stats(self, logits_real, pool3_real, logits_gen, pool3_gen):
    self.all_logits_real.append(logits_real)
    self.all_pool3_real.append(pool3_real)
    self.all_logits_gen.append(logits_gen)
    self.all_pool3_gen.append(pool3_gen)

  def reset(self):
    self.all_logits_real.clear()
    self.all_pool3_real.clear()
    self.all_logits_gen.clear()
    self.all_pool3_gen.clear()
    return

  def compute_fid_score(self):
    """Return a dict of metrics."""
    metrics = {}
    logging.info("Computing Inception score.")
    all_logits_gen = np.concatenate(self.all_logits_gen, axis=0)
    logging.info("IS number of gen samples: %d, number of classes: %d",
                 all_logits_gen.shape[0], all_logits_gen.shape[1])
    is_score = tfgan.eval.classifier_score_from_logits(all_logits_gen)
    metrics.update({"inception_score": is_score})
    logging.info("Computing FID score.")
    all_stats_real = np.concatenate(self.all_pool3_real, axis=0)
    all_stats_gen = np.concatenate(self.all_pool3_gen, axis=0)
    logging.info("FID number of real samples: %d", all_stats_real.shape[0])
    logging.info("FID number of generated samples: %d", all_stats_gen.shape[0])
    gen_mean, gen_cov = get_stats_for_fid(all_stats_gen)
    ref_mean, ref_cov = get_stats_for_fid(all_stats_real)
    metrics.update({
        "fid_batch": get_fid_score(gen_mean, gen_cov, ref_mean, ref_cov),
    })
    if self.dataset_stats_mean is not None:
      metrics.update({
          "fid_full":
              get_fid_score(gen_mean, gen_cov, self.dataset_stats_mean,
                            self.dataset_stats_cov),
          "fid_batch_vs_full":
              get_fid_score(ref_mean, ref_cov, self.dataset_stats_mean,
                            self.dataset_stats_cov),
      })
    return metrics
