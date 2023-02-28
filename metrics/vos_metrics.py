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
"""Video object segmentation metrics."""

import io
import os
import tempfile

from absl import logging
from davis2017.evaluation import DAVISEvaluation
import numpy as np
import pandas as pd
import PIL
import utils
from metrics import metric_registry
import tensorflow as tf

_PALETTE = [
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128,
    128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0,
    128, 191, 0, 128, 64, 128, 128, 191, 128, 128
]


@metric_registry.MetricRegistry.register('davis_video_object_segmentation')
class DavisVideoObjectSegmentationMetric():
  """Video object segmentation metric for DAVIS."""

  def __init__(self, config):
    self.config = config
    self.metric_names = [
        'J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall',
        'F-Decay'
    ]
    self.results_dir = config.task.metric.get('results_dir')
    self.dataset_eval = DAVISEvaluation(
        davis_root=config.dataset.annotations_dir,
        task=config.dataset.vos_task,
        gt_set=('train' if config.dataset.eval_split == 'train' else 'val'),
        use_tfds=False)
    self.reset_states()

  def reset_states(self):
    self.metric_values = None
    self._local_pred_dir_obj = tempfile.TemporaryDirectory()

  def record_prediction(self, predictions, video_name, frame_ids, step):
    """Records predictions.

    Args:
      predictions: uint8 of shape (num_frames, h, w, channel), where channel
        could be 1 or >1, but only the last channel is used as instance id.
      video_name: str. Video name.
      frame_ids: list of int, or 1-d np.array. Frame ids of predictions.
      step: int. The checkpoint step, used to name sub-directories.
    """
    predictions = predictions[..., -1]  # Last channel is instance id.
    subdir = os.path.join(self._local_pred_dir_obj.name, str(step), video_name)
    if not tf.io.gfile.exists(subdir):
      tf.io.gfile.makedirs(subdir)

    for frame_id in frame_ids:
      filename = f'{frame_id:05}.png'
      filepath = os.path.join(subdir, filename)
      pred_image = PIL.Image.fromarray(predictions[frame_id], mode='L')
      pred_image.putpalette(_PALETTE)
      with io.BytesIO() as out:
        pred_image.save(out, format='PNG')
        with tf.io.gfile.GFile(filepath, 'wb') as f:
          f.write(out.getvalue())
    logging.info('Done writing out pngs for %s', video_name)

    if self.results_dir is not None:
      # Copy images to results dir.
      results_dir = os.path.join(self.results_dir, str(step), video_name)
      utils.copy_dir(subdir, results_dir)

  def _evaluate(self, step):
    """Evaluates with predictions for all images.

    Call this function from `self.result`.

    Args:
      step: int. The checkpoint step being evaluated.

    Returns:
      dict from metric name to float value.
    """
    result_path = os.path.join(self._local_pred_dir_obj.name, str(step))
    metrics_res = self.dataset_eval.evaluate(result_path)
    J, F = metrics_res['J'], metrics_res['F']  # pylint: disable=invalid-name
    g_measures = [
        'J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall',
        'F-Decay'
    ]
    g_res = np.array([
        (np.mean(J['M']) + np.mean(F['M'])) / 2.,
        np.mean(J['M']),
        np.mean(J['R']),
        np.mean(J['D']),
        np.mean(F['M']),
        np.mean(F['R']),
        np.mean(F['D'])
    ])

    if self.results_dir is not None:
      # Write metrics to result_dir.
      result_dir = os.path.join(self.results_dir, str(step))
      csv_name_global_path = os.path.join(result_dir, 'global_results.csv')
      csv_name_per_sequence_path = os.path.join(result_dir,
                                                'per_sequence_results.csv')

      # Global results.
      g_res_ = np.reshape(g_res, [1, len(g_res)])
      table_g = pd.DataFrame(data=g_res_, columns=g_measures)
      with tf.io.gfile.GFile(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format='%.3f')
      logging.info('Global results saved in %s', csv_name_global_path)

      # Per sequence results.
      assert isinstance(J['M_per_object'], dict)
      seq_names = list(J['M_per_object'].keys())
      seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
      j_per_object = [J['M_per_object'][x] for x in seq_names]
      f_per_object = [F['M_per_object'][x] for x in seq_names]
      table_seq = pd.DataFrame(
          data=list(zip(seq_names, j_per_object, f_per_object)),
          columns=seq_measures)
      with tf.io.gfile.GFile(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format='%.3f')
      logging.info('Per-sequence results saved in %s',
                   csv_name_per_sequence_path)

    return {name: v for name, v in zip(g_measures, g_res)}

  def result(self, step):
    """Return the metric values (and compute it if needed)."""
    if self.metric_values is None:
      self.metric_values = self._evaluate(step)
    return self.metric_values
