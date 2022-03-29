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
"""Common task utils."""

import json
import os
import utils
import vocab
import tensorflow as tf


def coco_annotation_path(config, ret_category_names=True):
  """Returns coco annotation path and category names (optionally)."""
  gt_annotations_path = None
  category_names = {}
  if config.dataset.get('coco_annotations_dir'):
    split = config.dataset.train_split if config.training else (
        config.dataset.eval_split)
    filename = (
        config.dataset.train_filename
        if split == 'train' else config.dataset.val_filename)
    gt_annotations_path = os.path.join(config.dataset.coco_annotations_dir,
                                       filename)
    if ret_category_names:
      with tf.io.gfile.GFile(gt_annotations_path, 'r') as f:
        annotations = json.load(f)
      category_names = {c['id']: c for c in annotations['categories']}
  return {
      'gt_annotations_path': gt_annotations_path,
      'category_names': category_names
  }


def build_prompt_seq_from_task_id(task_vocab_id: int,
                                  response_seq=None,
                                  prompt_shape=None):
  """"Build prompt seq just using task id.

  Args:
    task_vocab_id: Vocab id for the task.
    response_seq: an (optional) discerte target sequen with shape (bsz, ..., k).
    prompt_shape: an (optional) tuple for prompt shape. One and only one of
      `response_seq` and `prompt_shape` should be specified.

  Returns:
    discrete input sequence of task id with shape (bsz, ..., 1).
  """
  task_id = tf.constant(task_vocab_id)
  if response_seq is not None:
    prompt_seq = tf.zeros_like(response_seq[..., :1]) + tf.cast(
        task_id, response_seq.dtype)
  if prompt_shape is not None:
    assert response_seq is None, 'double specification'
    prompt_seq = tf.zeros(prompt_shape, dtype=tf.int64) + tf.cast(
        task_id, dtype=tf.int64)
  return prompt_seq


def decode_object_seq_to_bbox(logits,
                              pred_seq,
                              quantization_bins,
                              coord_vocab_shift):
  """Decode objects (label & bbox) for seq from `build_response_seq_from_bbox`.

  Assume yxyxc format with truncation at the end for any uneven extra tokens.
    Replace class tokens with argmax instead of sampling.

  Args:
    logits: `float` output logits in shape of (bsz, max_seq_len, vocab_size).
    pred_seq: `int` pred sequence in shape of (bsz, max_seq_len).
    quantization_bins: `int` for bins.
    coord_vocab_shift: `int`, shifting coordinates by a specified integer.

  Returns:
    pred_class: `int` of shape (bsz, max_instances_per_image).
    pred_bbox: `float` of shape (bsz, max_instances_per_image, 4).
    pred_score: `float` of shape (bsz, max_instances_per_image).
  """
  _, seqlen, vocab_size = logits.shape
  if seqlen % 5 != 0:  # truncate out the last few tokens.
    pred_seq = pred_seq[..., :-(seqlen % 5)]
    logits = logits[..., :-(seqlen % 5), :]
  pred_class_p = tf.nn.softmax(logits)[:, 4::5]  # (bsz, instances, vocab_size)
  mask_s1 = [0.] * vocab.BASE_VOCAB_SHIFT  # reserved.
  mask_s2 = [1.] * (coord_vocab_shift - vocab.BASE_VOCAB_SHIFT)  # labels.
  mask_s3 = [0] * (vocab_size - coord_vocab_shift)  # coordinates and others.
  mask = tf.constant(mask_s1 + mask_s2 + mask_s3)
  pred_class = tf.argmax(pred_class_p * mask[tf.newaxis, tf.newaxis, :], -1)
  pred_score = tf.reduce_sum(
      pred_class_p * tf.one_hot(pred_class, vocab_size), -1)
  pred_class = tf.maximum(pred_class - vocab.BASE_VOCAB_SHIFT, 0)
  pred_bbox = seq_to_bbox(pred_seq - coord_vocab_shift, quantization_bins)
  return pred_class, pred_bbox, pred_score


def seq_to_bbox(seq, quantization_bins, seq_format='yxyx_name'):
  """Returns [0, 1] normalized yxyx bbox from token sequence."""
  # [batch, 5*num_instances]
  assert seq.shape.rank == 2, seq.shape.as_list()
  # [batch, num_instances, 1]
  if seq_format.startswith('name'):
    ymin = tf.expand_dims(seq[:, 1::5], -1)
    xmin = tf.expand_dims(seq[:, 2::5], -1)
    ymax = tf.expand_dims(seq[:, 3::5], -1)
    xmax = tf.expand_dims(seq[:, 4::5], -1)
  else:
    ymin = tf.expand_dims(seq[:, 0::5], -1)
    xmin = tf.expand_dims(seq[:, 1::5], -1)
    ymax = tf.expand_dims(seq[:, 2::5], -1)
    xmax = tf.expand_dims(seq[:, 3::5], -1)
  if seq_format in ['name_cycxhw', 'cycxhw_name']:
    ycnt, xcnt, ysize, xsize = ymin, xmin, ymax, xmax
    ymin = ycnt - ysize//2
    xmin = xcnt - xsize//2
    ymax = ycnt + ysize//2
    xmax = xcnt + xsize//2
  quantized_box = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
  quantized_box = utils.dequantize(quantized_box, quantization_bins)
  return tf.minimum(tf.maximum(quantized_box, 0), 1)


def join_if_not_none(args, sep):
  args = [str(arg) for arg in args if arg is not None]
  return sep.join(args)

