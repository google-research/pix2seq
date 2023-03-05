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
"""Video panoptic segmentation config."""

# pylint: disable=invalid-name,line-too-long

from configs import config_diffusion_panoptic_base as config_base
from configs import transform_configs


def get_config(config_str=None):
  """Returns config."""
  if config_str:
    task_variant = config_str
  else:
    task_variant = 'video_panoptic_segmentation@kittistep_vps'

  encoder_variant = 'resnet-c'
  decoder_variant = 'transunet'

  if 'kittistep' in task_variant:
    imgsize = '384x1248'
    msize = '192x624'
  elif 'davis' in task_variant:
    imgsize = '512x1024'
    msize = '256x512'
  else:
    imgsize = '256x256'
    msize = '128x128'

  config = config_base.get_config(
      f'{task_variant},{encoder_variant},{decoder_variant},{imgsize},{msize}')

  image_size = [int(x) for x in imgsize.split('x')]
  mask_size = [int(x) for x in msize.split('x')]
  config.task.train_transforms = transform_configs.get_video_panoptic_segmentation_train_transforms(
      image_size, mask_size, 1.0, 1.0, 0.)
  config.task.eval_transforms = transform_configs.get_video_panoptic_segmentation_eval_transforms(
      image_size, mask_size, 100)

  config.model.name = 'panoptic_diffusion'
  config.model.train_schedule = 'cosine'
  config.model.l_tile_factors = 1
  config.model.frozen_backbone = False
  config.model.enc_drop = 0.
  config.model.enc_fuse = 'pyramid_merge'
  config.model.enc_fuse_upsample = 'nearest'
  config.model.enc_fuse_dim = 256
  config.model.b_scale = 0.1
  config.model.pred_type = 'x_softmax_xent'
  config.model.self_cond = 'none'
  config.model.conditional = 'cat+attn'
  config.model.mask_weight_p = 0.2

  config.decoder.mhsa_resolutions = '0'
  config.decoder.n_mlp_blocks = 0
  config.decoder.in_kernel_size = 1
  config.decoder.out_kernel_size = 1
  config.decoder.output_residual = False
  config.decoder.input_scaling = False
  config.decoder.dim = 128
  config.decoder.udrop = 0.
  config.decoder.n_res_blocks = '1,1,1,1'
  config.decoder.ch_multipliers = '1,1,2,2'
  config.decoder.transformer_strides = 1
  config.decoder.transformer_dim = 512
  config.decoder.transformer_blocks = 6
  config.decoder.outp_softmax_groups = 2

  config.optimization.learning_rate_schedule = 'linear'
  config.optimization.end_lr_factor = 0.02
  config.optimization.weight_decay = 0.05
  config.optimization.beta2 = 0.999
  config.optimization.warmup_epochs = 0
  config.optimization.global_clipnorm = 1.

  config.task.proceeding_frames = '-2,-1'
  config.task.eval_single_frames = False
  config.task.eval_use_gt_cond_frames = False

  if 'davis' in task_variant:
    config.task.max_instances_per_image = 16
    config.task.max_num_frames = 105
    config.task.eval_transforms[4].max_num_frames = 105
    config.task.metric.name = 'davis_video_object_segmentation'

  config.eval.batch_size = 2
  return config


def get_sweep(h):
  """Get the hyperparamater sweep."""

  return h.chainit([
      h.product([
          h.sweep('config.train.steps', [50000]),
          h.sweep('config.train.batch_size', [32]),
          h.sweep('config.train.checkpoint_steps', [1000]),
          h.sweep('config.optimization.learning_rate', [1e-5, 3e-5]),
          h.sweep('config.optimization.end_lr_factor', [1.]),
          h.sweep('config.optimization.warmup_epochs', [0]),
          h.sweep('config.optimization.ema_decay', [0.99]),
          h.zipit([
              h.sweep('config.task.train_transforms[0].min_scale', [1.0]),
              h.sweep('config.task.train_transforms[0].max_scale', [1.0]),
          ]),
          h.sweep('config.task.train_transforms[3].color_jitter_strength', [0.]),
          h.sweep('config.task.object_order', ['shuffle']),
          h.sweep('config.task.frames_dropout', [0.2]),
      ]),
  ])


