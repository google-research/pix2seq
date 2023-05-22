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
"""A config."""

# pylint: disable=invalid-name,line-too-long

from configs.google.users.iamtingchen import config_diffusion_base as config_base

DATA_NAME = 'cifar10'
ARCH_VARIANT = 'transunet'


def get_config(config_str=None):
  """Returns config."""
  del config_str
  config = config_base.get_config(f'{DATA_NAME},{ARCH_VARIANT}')
  config.model.name = 'image_discrete_diffusion_model'
  config.model.b_scale = 1.0
  config.model.pred_type = 'x'
  config.model.conditional = 'class'
  config.optimization.ema_decay = 0.9999
  config.eval.batch_size = 80
  config.eval.steps = 625
  return config


def get_sweep(h):
  """Get the hyperparamater sweep."""
  if ARCH_VARIANT == 'transunet':
    return h.chainit([
        h.product([
            h.sweep('config.train.steps', [1_500_000]),
            h.sweep('config.train.checkpoint_steps', [10000]),
            h.sweep('config.train.keep_checkpoint_max', [100]),
            h.sweep('config.train.batch_size', [128*1]),
            h.sweep('config.optimization.learning_rate', [1e-4]),
            h.sweep('config.optimization.warmup_steps', [10000]),
            h.sweep('config.optimization.optimizer', ['adamw']),
            h.sweep('config.optimization.exclude_from_weight_decay', ['']),
            h.sweep('config.model.self_cond', ['x']),
            h.sweep('config.model.self_cond_by_masking', [False]),
            h.sweep('config.model.self_cond_rate', [0.5]),
            h.sweep('config.model.b_scale', [0.5]),
            h.sweep('config.model.udrop', [0.]),
            h.sweep('config.model.dim', [256]),
            h.sweep('config.model.n_res_blocks', ['3,3,3']),
            h.sweep('config.model.ch_multipliers', ['1,1,1']),
            h.sweep('config.model.total_time_steps', [1000]),
            h.sweep('config.model.pred_type', ['x_softmax_xent']),
            h.zipit([
                h.sweep('config.model.b_type', ['uint8', 'uint8_s']),
                h.sweep('config.model.outp_softmax_groups', [0, 3]),
            ]),
        ]),
    ])


def get_eval_args_and_tags(config, args, unused_config_flag):
  """Return eval args and tags."""
  args_and_tags = []
  for eval_split in [config.dataset.train_split]:
    for sampler_name in ['ddpm']:
      for infer_schedule in ['cosine']:
        for infer_iterations in [100, 250, 400]:
          # if sampler_name == 'ddim' and infer_iterations > 250: continue
          # if sampler_name == 'ddpm' and infer_iterations <= 250: continue
          eval_args = args.copy()
          sampler_name_s = sampler_name.replace('@', '')
          infer_schedule_s = infer_schedule.replace('@', '').replace(',', 'c')
          eval_tag = f'ev_{eval_split}_{sampler_name_s}_{infer_schedule_s}_i{infer_iterations}'
          eval_args.update({
              'config.eval.tag': eval_tag,
              'config.dataset.eval_split': eval_split,
              'config.model.sampler_name': sampler_name,
              'config.model.infer_schedule': infer_schedule,
              'config.model.infer_iterations': infer_iterations,
          })
          args_and_tags.append((eval_args, eval_tag, None))
  return args_and_tags
