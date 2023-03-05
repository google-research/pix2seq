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
"""Config for conditional mask modeling."""

# pylint: disable=invalid-name,line-too-long

from configs import config_diffusion_panoptic_base as config_base
from configs import transform_configs

IMAGE_SIZE = '1024x1024'
MASK_SIZE = '512x512'
MODE = 'train_high_res'  # one of ['train_high_res', 'train_low_res']


def get_config(config_str=None):
  """Returns config."""
  if config_str:
    task_variant = config_str
  else:
    task_variant = 'panoptic_segmentation@coco/2017_panoptic_segmentation'
  encoder_variant = 'resnet-c'
  decoder_variant = 'transunet'
  config = config_base.get_config(
      f'{task_variant},{encoder_variant},{decoder_variant},{IMAGE_SIZE},{MASK_SIZE}')
  image_size = [int(x) for x in IMAGE_SIZE.split('x')]
  mask_size = [int(x) for x in MASK_SIZE.split('x')]
  config.task.train_transforms = transform_configs.get_panoptic_segmentation_train_transforms(
      image_size, mask_size, 1.0, 1.0, 0.)
  config.task.eval_transforms = transform_configs.get_panoptic_segmentation_eval_transforms(
      image_size)
  config.model.name = 'panoptic_diffusion'
  config.model.train_schedule = 'cosine'
  config.model.l_tile_factors = 1
  config.model.frozen_backbone = False
  config.model.enc_drop = 0.
  config.model.enc_fuse = 'pyramid_merge'
  config.model.enc_fuse_upsample = 'nearest'
  config.model.enc_fuse_dim = 256
  config.model.total_time_steps = 1.0  # for legacy compability.
  config.decoder.mhsa_resolutions = '0'
  config.decoder.n_mlp_blocks = 0
  config.decoder.outp_softmax_groups = 0
  config.decoder.in_kernel_size = 1
  config.decoder.out_kernel_size = 1
  config.optimization.learning_rate_schedule = 'linear'
  config.optimization.end_lr_factor = 0.02
  config.optimization.weight_decay = 0.05
  config.optimization.beta2 = 0.999
  config.optimization.warmup_epochs = 5
  config.optimization.global_clipnorm = 1.
  return config


def get_sweep(h):
  """Get the hyperparamater sweep."""
  if MODE == 'train_low_res':
    return h.chainit([
        h.product([
            h.sweep('config.train.epochs', [800]),
            h.sweep('config.train.batch_size', [512]),
            h.sweep('config.train.checkpoint_epochs', [10]),
            h.sweep('config.train.keep_checkpoint_max', [10]),
            h.sweep('config.optimization.learning_rate', [1e-4]),
            h.sweep('config.optimization.end_lr_factor', [0.1]),
            h.sweep('config.optimization.warmup_epochs', [5]),
            h.sweep('config.optimization.ema_decay', [0.999]),
            h.sweep('config.model.b_scale', [0.1]),
            h.sweep('config.model.pred_type', ['x_softmax_xent']),
            h.sweep('config.model.self_cond', ['none']),
            h.sweep('config.model.conditional', ['cat+attn']),
            h.sweep('config.model.mask_weight_p', [0.2]),
            h.sweep('config.task.train_transforms[1].min_scale', [1.0]),  # jitter_scale
            h.sweep('config.task.train_transforms[1].max_scale', [3.0]),
            h.sweep('config.task.train_transforms[4].color_jitter_strength', [1.0]),
            h.sweep('config.decoder.dim', [128]),
            h.sweep('config.decoder.udrop', [0.]),
            h.sweep('config.decoder.n_res_blocks', ['1,1,1,1']),
            h.sweep('config.decoder.ch_multipliers', ['1,1,2,2']),
            h.sweep('config.decoder.transformer_strides', [1]),
            h.sweep('config.decoder.transformer_dim', [512]),
            h.sweep('config.decoder.transformer_blocks', [6]),
            h.sweep('config.decoder.outp_softmax_groups', [2]),
        ]),
    ])
  elif MODE == 'train_high_res':
    return h.chainit([
        h.product([
            h.sweep('config.train.epochs', [15]),
            h.sweep('config.train.batch_size', [16]),
            h.sweep('config.train.checkpoint_epochs', [1]),
            h.sweep('config.optimization.learning_rate', [1e-5]),
            h.sweep('config.optimization.end_lr_factor', [0.1]),
            h.sweep('config.optimization.warmup_epochs', [0]),
            h.sweep('config.optimization.ema_decay', [0.999]),
            h.sweep('config.model.b_scale', [0.1]),
            h.sweep('config.model.pred_type', ['x_softmax_xent']),
            h.sweep('config.model.self_cond', ['none']),
            h.sweep('config.model.conditional', ['cat+attn']),
            h.sweep('config.model.mask_weight_p', [0.2]),
            h.sweep('config.task.train_transforms[1].min_scale', [1.0]),
            h.sweep('config.task.train_transforms[1].max_scale', [1.0]),
            h.sweep('config.decoder.dim', [128]),
            h.sweep('config.decoder.udrop', [0.]),
            h.sweep('config.decoder.n_res_blocks', ['1,1,1,1']),
            h.sweep('config.decoder.ch_multipliers', ['1,1,2,2']),
            h.sweep('config.decoder.transformer_strides', [1]),
            h.sweep('config.decoder.transformer_dim', [512]),
            h.sweep('config.decoder.transformer_blocks', [6]),
            h.sweep('config.decoder.outp_softmax_groups', [2]),
        ]),
    ])


def get_eval_args_and_tags(config, args, unused_config_flag):
  """Return eval args and tags."""
  args_and_tags = []
  for eval_split in [config.dataset.eval_split]:
    for sampler in ['ddim']:
      for iterations in [20]:
        for td in [1.0, 2.0]:
          for min_pixels in [40]:
            eval_args = args.copy()
            eval_tag = f'ev_{eval_split}_{sampler}_i{iterations}_td{td}_p{min_pixels}'
            results_dir = eval_args['model_dir'] + '/' + eval_tag  # pylint: disable=unused-variable
            eval_args.update({
                'config.eval.tag': eval_tag,
                'config.eval.batch_size': 8,
                'config.eval.steps': 0,
                'config.model.sampler': sampler,
                'config.model.iterations': iterations,
                'config.model.td': td,
                'config.task.min_pixels': min_pixels,
                # 'config.task.metric.results_dir': results_dir,
            })
            if eval_split == 'train':
              eval_args.update({
                  'config.dataset.eval_split': 'train',
                  'config.eval.steps': 100,
              })
            args_and_tags.append((eval_args, eval_tag, None))
  return args_and_tags


