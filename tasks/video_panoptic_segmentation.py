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
"""Video panoptic segmentation task."""

from absl import logging
import ml_collections
import utils
from data import video as video_datasets
from metrics import metric_registry
from tasks import task as task_lib
from tasks import task_utils
import tensorflow as tf


@task_lib.TaskRegistry.register('video_panoptic_segmentation')
class TaskVideoPanopticSegmentation(task_lib.Task):
  """Video panoptic segmentation task."""

  def __init__(self,
               config: ml_collections.ConfigDict):
    super().__init__(config)
    metric_config = config.task.get('metric')
    if metric_config and metric_config.get('name'):
      self._metrics = metric_registry.MetricRegistry.lookup(metric_config.name)(
          config)

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
    t_cfg = self.config.task

    def _convert_video_to_image_features(example):
      new_example = {
          'orig_image_size': tf.shape(example['video/frames'])[1:3],
          'video_id': example['video/id'],
          'num_frames': example['video/num_frames'],
      }
      label_map = example['label_map']

      if training or t_cfg.eval_single_frames:
        new_example['image'] = example['video/frames'][-1]
        new_example['label_map'] = label_map[-1]

        # Get conditional maps.
        num_frames = self.config.dataset.num_frames
        assert num_frames > 1, 'There should be at least 2 frames in example.'
        assert len(t_cfg.proceeding_frames.split(',')) == num_frames - 1
        proceeding_masks = []
        for i in range(num_frames - 1):
          keep = tf.random.uniform(
              []) > t_cfg.frames_dropout if training else True
          proceeding_masks.append(
              tf.cond(keep, lambda: label_map[i],  # pylint: disable=cell-var-from-loop
                      lambda: tf.zeros_like(new_example['label_map'])))
          # [h, v, num_proc_frames * 2]
        new_example['cond_map'] = tf.concat(proceeding_masks, axis=-1)
      else:
        new_example['image'] = example['video/frames']
        new_example['cond_map'] = label_map
        new_example['label_map'] = label_map
      return new_example

    dataset = dataset.map(_convert_video_to_image_features,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x: self.preprocess_single_example(x, training, batch_duplicates),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def preprocess_batched(self, batched_examples, training):
    """Task-specific preprocessing of batched examples on accelerators (TPUs).

    Args:
      batched_examples: tuples of feature and label tensors that are
        preprocessed, batched, and stored with `dict`.
      training: bool.

    Returns:
      images: `float` of shape (bsz, h, w, c) or (bsz, num_frames, h, w, c).
      label_map: `int32` of shape (bsz, h, w, 2), or when evaluating the entire
          video, of shape (bsz, num_frames, h, w, 1).
    """
    t_cfg = self.config.task
    m_cfg = self.config.model
    num_cond_frames = self.config.dataset.num_frames - 1

    if training or t_cfg.eval_single_frames:
      mask_weight = task_utils.get_normalized_weight(
          batched_examples['label_map'][..., 1],
          t_cfg.max_instances_per_image,
          m_cfg.mask_weight_p)
      label_map = task_utils.integer_map_to_bits(
          batched_examples['label_map'], t_cfg.n_bits_label, m_cfg.b_scale)
      cond_map = task_utils.integer_map_to_bits(
          batched_examples['cond_map'], t_cfg.n_bits_label * num_cond_frames,
          m_cfg.b_scale, num_channels=num_cond_frames * 2)
    else:  # eval the entire video
      label_map = batched_examples['label_map']  # [bsz, num_frames, h, w, 2]
      cond_map = batched_examples['label_map']

    if training:
      return (batched_examples['image'], cond_map), label_map, mask_weight
    else:
      return (batched_examples['image'], cond_map), label_map, batched_examples

  def infer(self, model, preprocessed_outputs):
    """Perform  given the model and preprocessed outputs."""
    t_cfg = self.config.task
    m_cfg = self.config.model
    images, label_map, examples = preprocessed_outputs

    if t_cfg.eval_single_frames:
      masks_pred = label_map  # comment `model.infer` for sanity check
      masks_pred = model.infer(images, m_cfg.iterations, m_cfg.sampler)
      masks_pred = task_utils.bits_to_panoptic_map(
          masks_pred, t_cfg.n_bits_label, self.config.dataset.num_classes,
          t_cfg. max_instances_per_image)
      if m_cfg.image_size != m_cfg.msize:
        masks_pred = tf.image.resize(
            masks_pred, m_cfg.image_size, method='nearest')
    else:  # eval the entire video
      masks_pred = self.infer_video(model, images)
    return examples, masks_pred

  def infer_video(self, model, images):
    t_cfg = self.config.task
    m_cfg = self.config.model
    frames, cond_map = images
    num_frames = t_cfg.max_num_frames
    mh, mw = m_cfg.msize
    bsz = tf.shape(frames)[0]
    frames = tf.transpose(frames, [1, 0, 2, 3, 4])
    cond_map = tf.transpose(cond_map, [1, 0, 2, 3, 4])

    proceeding_frames = [
        int(s) for s in t_cfg.proceeding_frames.split(',')]
    num_cond_frames = len(proceeding_frames)

    def loop_body(masks_pred, step):
      frames_i = frames[step]

      # Get conditional masks.
      cond_maps = cond_map if t_cfg.eval_use_gt_cond_frames else masks_pred
      cond_map_i = []
      if proceeding_frames:
        for offset in proceeding_frames:
          cond_map_i.append(
              tf.cond(step + offset >= 0,
                      lambda: cond_maps[step + offset],  # pylint: disable=cell-var-from-loop
                      lambda: tf.zeros_like(cond_maps[0])))
        cond_map_i = tf.concat(cond_map_i, axis=-1)
      else:
        cond_map_i = tf.zeros_like(cond_maps[0])

      cond_map_i = task_utils.integer_map_to_bits(
          cond_map_i, t_cfg.n_bits_label * num_cond_frames, m_cfg.b_scale,
          num_channels=num_cond_frames * 2)

      # Inference on the current frame.
      if step == 0:
        masks_pred_o = model.infer(  # [bsz, h, w, 16]
            (frames_i, cond_map_i), m_cfg.iterations, m_cfg.sampler)
      else:
        masks_pred_o = model.infer(  # [bsz, h, w, 16]
            (frames_i, cond_map_i), m_cfg.iterations_2, m_cfg.sampler)

      masks_pred_o = task_utils.bits_to_panoptic_map(
          masks_pred_o, t_cfg.n_bits_label, self.config.dataset.num_classes,
          t_cfg.max_instances_per_image)

      if m_cfg.image_size[0] != m_cfg.msize[0] or (
          m_cfg.image_size[1] != m_cfg.msize[1]):
        masks_pred_o = tf.image.resize(
            masks_pred_o, m_cfg.msize, method='nearest')

      masks_pred = tf.tensor_scatter_nd_update(
          masks_pred, [[step]], [masks_pred_o])
      return masks_pred, step + 1

    def cond(masks_pred, step):
      del masks_pred
      return tf.less(step, num_frames)

    masks_pred_var = tf.zeros([num_frames, bsz, mh, mw, 2], tf.int32)
    step = 0
    masks_pred_var, step = tf.while_loop(
        cond=cond, body=loop_body, loop_vars=[masks_pred_var, step])
    return tf.transpose(masks_pred_var, [1, 0, 2, 3, 4])

  def postprocess_tpu(self, batched_examples, predictions, training=False):
    """Organizing results after fitting the batched examples in graph.

    Such as updating metrics, putting together results for computing metrics in
      CPU/numpy mode.

    Args:
      batched_examples: a tuple of features (`dict`) and labels (`dict`),
          containing images and labels.
      predictions: `int32` of shape (bsz, h, w, 2).
      training: bool.

    Returns:
      results for passing to `postprocess_cpu` which runs in CPU mode.
    """
    examples = batched_examples
    cond_map = examples['cond_map'] if 'cond_map' in examples else (
        tf.zeros_like(examples['num_frames']))

    return (examples['image'], cond_map, examples['video_id'],
            examples['orig_image_size'], examples['unpadded_image_size'],
            examples['num_frames'], predictions, examples['label_map'])

  def postprocess_cpu(self, outputs, train_step,
                      eval_step=None, training=False, summary_tag='eval',
                      ret_results=False):
    """CPU post-processing of outputs.

    Such as computing the metrics, log image summary.

    Args:
      outputs: a tuple of tensor passed from `postprocess_tpu`.
      train_step: `int` scalar indicating training step of current model or
          the checkpoint.
      eval_step: `int` scalar indicating eval step for the given checkpoint.
      training: `bool` indicating training or  mode.
      summary_tag: `string` of name scope for result summary.
      ret_results: whether to return visualization images.

    Returns:
      A dict of visualization images if ret_results, else None.
    """
    t_cfg = self.config.task
    m_cfg = self.config.model
    d_cfg = self.config.dataset

    # Copy outputs to cpu.
    new_outputs = []
    for i in range(len(outputs)):
      logging.info('Copying output at index %d to cpu for cpu post-process', i)
      new_outputs.append(tf.identity(outputs[i]))
    images, unused_cond_map, video_id, orig_image_sizes, unpadded_image_sizes, num_frames, predictions, gts = new_outputs  # pylint: disable=unbalanced-tuple-unpacking

    if t_cfg.eval_single_frames:
      if eval_step <= 15 or ret_results:
        # Image summary.
        preds_s = utils.colorize(
            predictions[..., 0], vmin=0, vmax=d_cfg.num_classes)
        preds_i = utils.colorize(
            predictions[..., 1], vmin=0, vmax=t_cfg.max_instances_per_image)
        gts_s = utils.colorize(
            gts[..., 0], vmin=0, vmax=d_cfg.num_classes)
        gts_i = utils.colorize(
            gts[..., 1], vmin=0, vmax=t_cfg.max_instances_per_image)

        new_image = tf.concat(
            [images, gts_s, preds_s, gts_i, preds_i], 1)
        tf.summary.image(summary_tag, new_image, step=train_step)

    else:
      vis_images = []
      frame_ids = [i * 10 for i in range(t_cfg.max_num_frames // 10)]
      logging.info('video_id: %s', video_id.numpy())
      for frame_id in frame_ids:
        im = images[:, frame_id]
        if m_cfg.image_size[0] != m_cfg.msize[0] or (
            m_cfg.image_size[1] != m_cfg.msize[1]):
          # Resize for visualization.
          im = tf.image.resize(im, m_cfg.msize,
                               method=tf.image.ResizeMethod.BICUBIC)
        gt_s = utils.colorize(gts[:, frame_id, ..., 0], vmin=0,
                              vmax=d_cfg.num_classes)
        pr_s = utils.colorize(predictions[:, frame_id, ..., 0], vmin=0,
                              vmax=d_cfg.num_classes)
        gt_i = utils.colorize(gts[:, frame_id, ..., 1], vmin=0,
                              vmax=t_cfg.max_instances_per_image)
        pr_i = utils.colorize(predictions[:, frame_id, ..., 1], vmin=0,
                              vmax=t_cfg.max_instances_per_image)
        vis_images.append(tf.concat([im, gt_s, pr_s, gt_i, pr_i], 2))

      if (eval_step <= 15) or ret_results:
        new_image = tf.concat(vis_images, 1)
        tf.summary.image(summary_tag, new_image, step=train_step)

      # Record predictions.
      if self._metrics is not None:
        self.record_prediction(
            predictions, video_id.numpy(), orig_image_sizes.numpy(),
            unpadded_image_sizes.numpy(), num_frames.numpy(), train_step)

    logging.info('Done post-process')
    if ret_results:
      return new_image

  def record_prediction(self, predictions, video_id, orig_image_sizes,
                        unpadded_image_sizes, num_frames, step):
    bsz = predictions.shape[0]
    max_frames = predictions.shape[1]
    mh, mw = self.config.model.msize
    imh, imw = self.config.model.image_size
    # Resize predicted masks to imsize, so that they can be converted to the
    # original image size later.
    predictions = tf.reshape(predictions, [bsz * max_frames, mh, mw, 2])
    predictions = tf.image.resize(predictions, [imh, imw],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    predictions = tf.reshape(predictions, [bsz, max_frames, imh, imw, 2])
    predictions = tf.cast(predictions, tf.uint8)
    for i in range(bsz):
      # Unpad and unresize images.
      unpadded_h, unpadded_w = unpadded_image_sizes[i]
      orig_h, orig_w = orig_image_sizes[i]
      pred_unpad = predictions[i, :, :unpadded_h, :unpadded_w]
      pred = tf.image.resize(
          pred_unpad, [orig_h, orig_w],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy()

      # TODO(lala) - Find a better way of converting video_id to video_name.
      if self.config.dataset.name == 'davis_vps':
        video_name = video_datasets.DavisDataset.VIDEO_NAMES[video_id[i]]
      elif self.config.dataset.name == 'kittistep_vps':
        video_name = f'{video_id[i]:04d}'
      else:
        video_name = str(video_id[i])
      self._metrics.record_prediction(pred, video_name, range(num_frames[i]),
                                      step)

  def compute_scalar_metrics(self, step):
    """Returns a dict containing scalar metrics to log."""
    if self.config.task.eval_single_frames or self._metrics is None:
      result = {}
    else:
      result = self._metrics.result(step)
    return result

  def reset_metrics(self):
    """Reset states of metrics accumulators."""
    if self._metrics is not None:
      self._metrics.reset_states()
