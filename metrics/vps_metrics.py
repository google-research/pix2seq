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
"""Video panoptic segmentation metrics."""

import collections
import io
import os
import tempfile
from typing import Sequence

from absl import logging
import numpy as np
import pandas as pd
import PIL
import utils
from metrics import metric_registry
from metrics import segmentation_and_tracking_quality as stq
import seaborn as sns
from skimage import segmentation
import tensorflow as tf

_SEMANTIC_PALETTE = [
    0, 0, 0,
    128, 0, 0,
    0, 128, 0,
    128, 128, 0,
    0, 0, 128,
    128, 0, 128,
    0, 128, 128,
    128, 128, 128,
    64, 0, 0,
    191, 0, 0,
    64, 128, 0,
    191, 128, 0,
    64, 0, 128,
    191, 0, 128,
    64, 128, 128,
    191, 128, 128,
    31, 119, 180,
    255, 127, 14,
    44, 160, 44,
    214, 39, 40,
    148, 103, 189,
    140, 86, 75,
    227, 119, 194,
    127, 127, 127]

# For instance map palette, get 32 colors from the seaborn palette, repeat
# 8 times to get a 256-color palette. Assign (0, 0, 0) as the first color.
_INSTANCE_PALETTE = [0, 0, 0] + (list(
    np.asarray(
        np.reshape(np.asarray(sns.color_palette('Spectral', 32)) * 255, [-1]),
        np.uint8)) * 8)[:-3]

_PANOPTIC_METRIC_OFFSET = 256 * 256 * 256


def _write_to_png_file(img, filepath):
  with io.BytesIO() as out:
    img.save(out, format='PNG')
    with tf.io.gfile.GFile(filepath, 'wb') as f:
      f.write(out.getvalue())


def _semantic_instance_maps_from_rgb(filename, rgb_instance_label_divisor=256):
  with tf.io.gfile.GFile(filename, 'rb') as f:
    panoptic_map = np.array(PIL.Image.open(f)).astype(np.int32)
  semantic_map = panoptic_map[:, :, 0]
  instance_map = (
      panoptic_map[:, :, 1] * rgb_instance_label_divisor +
      panoptic_map[:, :, 2])
  return semantic_map, instance_map


def _panoptic_map_from_semantic_instance_maps(semantic_map, instance_map,
                                              panoptic_label_divisor):
  return semantic_map * panoptic_label_divisor + instance_map


def _panoptic_map_from_rgb(filename, panoptic_label_divisor,
                           rgb_instance_label_divisor=256):
  """Loads a rgb format panoptic map from file and encode to single channel."""
  semantic_map, instance_map = _semantic_instance_maps_from_rgb(
      filename, rgb_instance_label_divisor)
  panoptic_map = _panoptic_map_from_semantic_instance_maps(
      semantic_map, instance_map, panoptic_label_divisor)
  return panoptic_map


def _panoptic_map_to_rgb(semantic_map, instance_map, filename,
                         rgb_instance_label_divisor=256):
  """Converts a panoptic map to rgb format and write to file."""
  instance_map_1 = instance_map // rgb_instance_label_divisor
  instance_map_2 = instance_map % rgb_instance_label_divisor
  panoptic_map = np.stack(
      [semantic_map, instance_map_1, instance_map_2], -1).astype(np.uint8)
  panoptic_map = PIL.Image.fromarray(panoptic_map)
  _write_to_png_file(panoptic_map, filename)


def _visualize_id_map(id_map, palette):
  boundaries = segmentation.find_boundaries(id_map, mode='thin')
  id_map[boundaries] = 0
  vis = PIL.Image.fromarray(id_map, mode='L')
  vis.putpalette(palette)
  return vis


def _get_new_id(exclusion_list, max_id):
  for i in range(max_id):
    if i not in exclusion_list:
      return i
  return 0


def _in_list_of_lists(x, list_of_lists):
  for l in list_of_lists:
    if x in l:
      return True
  return False


class STQEvaluation(object):
  """Evaluation class for the Segmentation and Tracking Quality (STQ)."""

  def __init__(self,
               annotation_dir: str,
               num_classes: int,
               class_has_instances_list: Sequence[int],
               ignore_label: int,
               panoptic_label_divisor: int,
               max_instances: int,
               num_cond_frames: int):
    self.stq = stq.STQuality(
        num_classes=num_classes,
        things_list=class_has_instances_list,
        ignore_label=ignore_label,
        max_instances_per_category=panoptic_label_divisor,
        offset=_PANOPTIC_METRIC_OFFSET)
    self.panoptic_label_divisor = panoptic_label_divisor
    self.annotation_dir = annotation_dir
    video_names = tf.io.gfile.listdir(annotation_dir)
    self.video_name_to_id_map = {
        video_names[i]: i  for i in range(len(video_names))
    }
    self.max_instances = max_instances
    self.num_cond_frames = num_cond_frames

  def evaluate(self, result_dir: str, postprocess_ins: bool = True):
    """Evaluates STQ for a result directory.

    Args:
      result_dir: str, directory that contains predictions.
      postprocess_ins: bool, whether to postprocess instance ids so that when
        new instances appear, they get assigned ids that have not been used in
        previous frames.

    Returns:
      a dict of metric name to value.
    """
    # Loop through all videos in result dir.
    for video_name in tf.io.gfile.listdir(result_dir):
      video_id = self.video_name_to_id_map[video_name]

      if postprocess_ins:
        # Keep a map of original instance ids to processed instance ids.
        id_map = np.zeros([self.max_instances], np.int32)
        # 'used_ids' is the list of all used ids. 'recent_ids' are ids that
        # appeared in the conditional frames. Only when an id appears in
        # previous frames other than the conditional frames, we need to map it
        # to a new id. Therefore we need to keep a running list of ids that
        # appear in the conditional frames.
        used_ids = [0]
        recent_ids = collections.deque([[0]] * self.num_cond_frames,
                                       self.num_cond_frames)

      for img in tf.io.gfile.listdir(os.path.join(result_dir, video_name)):
        pred_file = os.path.join(result_dir, video_name, img)
        gt_file = os.path.join(self.annotation_dir, video_name, img)
        gt = _panoptic_map_from_rgb(gt_file, self.panoptic_label_divisor)

        if postprocess_ins:
          semantic_map, instance_map = _semantic_instance_maps_from_rgb(
              pred_file, self.panoptic_label_divisor)

          # Update the id map.
          ids = list(np.unique(instance_map))
          for i in ids:
            if id_map[i] == 0:
              # the id hasn't appeared before.
              id_map[i] = i
              used_ids.append(i)
            elif _in_list_of_lists(i, recent_ids):
              # the id appeared in the conditional frames.
              pass
            else:
              # the id is not in the conditional frames, but has been used.
              new_id = _get_new_id(used_ids + ids, self.max_instances)
              id_map[i] = new_id
              used_ids.append(new_id)

          recent_ids.append(ids)
          # Update the instance map.
          instance_map = id_map[instance_map]

          pred = _panoptic_map_from_semantic_instance_maps(
              semantic_map, instance_map, self.panoptic_label_divisor)
        else:
          pred = _panoptic_map_from_rgb(pred_file, self.panoptic_label_divisor)
        self.stq.update_state(gt, pred, video_id)

    return self.stq.result()

  def reset_states(self):
    self.stq.reset_states()


@metric_registry.MetricRegistry.register('segmentation_and_tracking_quality')
class STQMetric(object):
  """Metric class for the Segmentation and Tracking Quality (STQ)."""

  def __init__(self, config):
    self.config = config
    self.results_dir = config.task.metric.get('results_dir')
    self.metric_names = ['AQ', 'IoU', 'STQ']
    self.per_sequence_metric_names = [
        'ID_per_seq', 'Length_per_seq', 'AQ_per_seq', 'IoU_per_seq', 
        'STQ_per_seq']
    self.eval = STQEvaluation(
        annotation_dir=config.dataset.annotations_dir,
        num_classes=config.dataset.num_classes - 1,  # exclude void pixel label
        class_has_instances_list=config.dataset.class_has_instances_list,
        ignore_label=config.dataset.ignore_label,
        panoptic_label_divisor=config.dataset.panoptic_label_divisor,
        max_instances=config.task.max_instances_per_image,
        num_cond_frames=len(config.task.proceeding_frames.split(',')))
    self.reset_states()

  def reset_states(self):
    self.metric_values = None
    self.eval.reset_states()
    # For saving predictions for metric evaluation.
    self._local_pred_dir_obj = tempfile.TemporaryDirectory()
    # For saving visualization images.
    self._panoptic_local_vis_dir_obj = tempfile.TemporaryDirectory()

  def _write_predictions(self, frame_id, semantic_map, instance_map, outdir):
    # When saving output for evaluation, change semantic ids back to the
    # original class ids, and 0 back to 255 which is to be ignored.
    semantic_map = np.where(semantic_map == 0,
                            self.config.dataset.ignore_label, semantic_map - 1)
    output_file = os.path.join(outdir, f'{frame_id:06}.png')
    _panoptic_map_to_rgb(semantic_map, instance_map, output_file)

  def _write_visualizations(self, frame_id, semantic_map, instance_map, outdir):
    """Writes visualization images to a directory.

    Args:
      frame_id: int, the frame id.
      semantic_map: uint8 of shape (h, w).
      instance_map: uint8 of shape (h, w).
      outdir: directory to write visualization images to.
    """
    sem = _visualize_id_map(semantic_map, _SEMANTIC_PALETTE)
    ins = _visualize_id_map(instance_map, _INSTANCE_PALETTE)

    _write_to_png_file(sem, os.path.join(outdir, f'{frame_id:06}_s.png'))
    _write_to_png_file(ins, os.path.join(outdir, f'{frame_id:06}_i.png'))

  def record_prediction(self, predictions, video_name, frame_ids, step):
    """Records predictions.

    Args:
      predictions: uint8 of shape (num_frames, h, w, 2), containing semantic
          map and instance map.
      video_name: str. Video name.
      frame_ids: list of int, or 1-d np.array. Frame ids of predictions.
      step: int. The checkpoint step, used to name sub-directories.
    """
    pred_dir = os.path.join(self._local_pred_dir_obj.name, str(step),
                            video_name)
    if not tf.io.gfile.exists(pred_dir):
      tf.io.gfile.makedirs(pred_dir)
    vis_dir = os.path.join(self._panoptic_local_vis_dir_obj.name, str(step),
                           video_name)
    if not tf.io.gfile.exists(vis_dir):
      tf.io.gfile.makedirs(vis_dir)

    # Write predictions to temporary directory.
    for fid, id_maps in zip(frame_ids, predictions):
      sem_map, ins_map = id_maps[..., 0], id_maps[..., 1]

      # Encode semantic map and instance_map into one image and write to file.
      self._write_predictions(fid, sem_map, ins_map, pred_dir)

      # Build visualization images and save to vis_dir.
      if self.results_dir is not None:
        self. _write_visualizations(fid, sem_map, ins_map, vis_dir)

    if self.results_dir is not None:
      # Copy visualization images to results dir.
      results_dir = os.path.join(self.results_dir, str(step), video_name)
      utils.copy_dir(vis_dir, results_dir)

      # TODO(lala) - delete this.
      results_dir = os.path.join(self.results_dir, 'pred', video_name)
      utils.copy_dir(pred_dir, results_dir)

    logging.info('Done writing out pngs for %s', video_name)

  def _evaluate(self, step):
    """Evaluates with predictions for all images.

    Call this function from `self.result`.

    Args:
      step: int. The checkpoint step being evaluated.

    Returns:
      dict from metric name to float value.
    """
    result_path = os.path.join(self._local_pred_dir_obj.name, str(step))
    stq_metric = self.eval.evaluate(result_path)

    if self.results_dir is not None:
      # Write metrics to result_dir.
      result_dir = os.path.join(self.results_dir, str(step))
      csv_name_global_path = os.path.join(result_dir, 'global_results.csv')
      csv_name_per_sequence_path = os.path.join(result_dir,
                                                'per_sequence_results.csv')

      # Global results.
      g_res = np.asarray([stq_metric[n] for n in self.metric_names])
      g_res_ = np.reshape(g_res, [1, len(g_res)])
      table_g = pd.DataFrame(data=g_res_, columns=self.metric_names)
      with tf.io.gfile.GFile(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format='%.3f')
      logging.info('Global results saved in %s', csv_name_global_path)

      # Per sequence results.
      table_seq = pd.DataFrame(
          data=list(zip(
              *[list(stq_metric[n]) for n in self.per_sequence_metric_names])),
          columns=self.per_sequence_metric_names)
      with tf.io.gfile.GFile(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format='%.3f')
      logging.info('Per-sequence results saved in %s',
                   csv_name_per_sequence_path)

    return {n: stq_metric[n] for n in self.metric_names}

  def result(self, step):
    """Return the metric values (and compute it if needed)."""
    if self.metric_values is None:
      self.metric_values = self._evaluate(step)
    return self.metric_values
