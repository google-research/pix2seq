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

import copy
# pylint: disable=invalid-name,line-too-long,missing-docstring
from configs import dataset_configs
from configs import transform_configs
from configs.config_base import architecture_config_map
from configs.config_base import D


def get_config(config_str=None):
  """config_str is either empty or contains task,architecture variants."""

  if config_str:
    config_lists = config_str.split(',')
    task_variant, encoder_variant, decoder_variant = config_lists[:3]
    image_dim, mdim = config_lists[3], config_lists[4]
    # The `if` conditions here are for backwards compatibility with existing
    # scripts that pass in a single dim for image size and mask size.
    # We should eventually remove this and require callers to pass in the full
    # res.
    if 'x' in image_dim:
      image_size = [int(d) for d in image_dim.split('x')]
    else:
      image_size = (int(image_dim), int(image_dim))
    if 'x' in mdim:
      msize = [int(d) for d in mdim.split('x')]
    else:
      msize = (int(mdim), int(mdim))
  else:
    task_variant = 'panoptic_segmentation@coco/2017_panoptic_segmentation'
    encoder_variant = 'resnet-c'
    decoder_variant = 'transunet'
    image_size = (256, 256)
    msize = (64, 64)

  tasks_and_datasets = []
  for task_and_ds in task_variant.split('+'):
    tasks_and_datasets.append(task_and_ds.split('@'))

  decoder_config_map = {
      'transformer': D(
          arch_name='transformer',
          num_layers=2,
          dim=512,
          dim_mlp=2048,
          num_heads=16,
          pos_encoding='learned',
          drop_path=0.0,
          drop_units=0.1,
          drop_att=0.0,
      ),
      'transunet': D(
          arch_name='transunet',
          dim=32,
          in_strides=1,
          in_kernel_size=1,
          out_kernel_size=1,
          udrop=0.1,
          n_mlp_blocks=0,
          n_res_blocks='1,1,1,1',
          kernel_sizes='3',
          ch_multipliers='1,2,3,4',
          transformer_dim=512,
          transformer_strides=1,
          transformer_blocks=1,
          mhsa_resolutions='16,8',
          per_head_dim=64,
          u_pos_encoding='sin_cos',
          outp_softmax_groups=16,
      ),
      'tape': D(
          arch_name='tape',
          num_layers='1,1',
          latent_slots=128,
          latent_dim=256*2,
          latent_mlp_ratio=4,
          latent_num_heads=16,
          tape_dim=256*2,
          tape_mlp_ratio=4,
          rw_num_heads=16,
          conv_kernel_size=0,
          conv_drop_units=0.,
          drop_path=0.,
          drop_units=0.,
          drop_att=0.,
          patch_size=8,
          outp_softmax_groups=2,
          pos_encoding='sin_cos',
          patch_scales='1',
          patch_scales_w='1',
          latent_pos_encoding='learned',
          tape_pos_encoding='learned',
      ),
  }

  task_config_map = {
      'panoptic_segmentation': D(
          name='panoptic_segmentation',
          vocab_id=16,
          image_size=image_size,
          n_bits_label=16,
          max_instances_per_image=101,
          object_order='random',
          color_jitter_strength=0.,
          jitter_scale_min=0.3,
          jitter_scale_max=1.0,
          min_pixels=40,
          weight=1.0,
          metric=D(
              name='coco_panoptic_segmentation',
              results_dir='',
          ),
      ),
      'video_panoptic_segmentation': D(
          name='video_panoptic_segmentation',
          vocab_id=18,
          image_size=image_size,
          n_bits_label=16,
          max_instances_per_image=256,  # including id 0.
          object_order='shuffle',
          color_jitter_strength=0.,
          jitter_scale_min=1.0,
          jitter_scale_max=1.0,
          weight=1.0,
          proceeding_frames='-2,-1',
          eval_single_frames=False,
          eval_use_gt_cond_frames=False,
          frames_dropout=0.1,
          max_num_frames=100,
          min_pixels=40,
          metric=D(
              name='segmentation_and_tracking_quality',
              results_dir=''
          ),
      ),
  }

  task_d_list = []
  dataset_list = []
  for task_name, ds_name in tasks_and_datasets:
    task_d_list.append(task_config_map[task_name])
    dataset_config = copy.deepcopy(dataset_configs.dataset_configs[ds_name])
    dataset_list.append(dataset_config)

  config = D(
      dataset=dataset_list[0],
      datasets=dataset_list,

      task=task_d_list[0],
      tasks=task_d_list,

      encoder=D(),

      decoder=D(),

      model=D(
          name='panoptic_diffusion',
          train_schedule='cosine',
          infer_schedule='cosine',
          train_noise='normal',
          infer_noise='normal',
          noise_std=1.0,
          noise_truncation=False,
          pred_type='x_softmax_xent',
          iter_start=0,
          step_bias=0.,
          td=0.,
          td_schedule='constant',
          x0_clip='auto',
          b_scale=0.1,
          normalize_noisy_input=False,
          total_time_steps=1000.,
          pretrained_ckpt='',
          self_cond='none',
          conditional='cat',
          iterations=100,
          iterations_2=100,  # only used in video inference where less iterations can be used for the 2nd frame onwards.
          sampler='ddim',
          l_tile_factors=1,
          msize=msize,
          mask_weight_p=0.,
          self_cond_rate=0.5,
          self_cond_by_masking=False,

          # extra architecture
          image_size=image_size,
          decoder_variant=decoder_variant,
          use_cls_token=False,
          patch_size=16,
          drop_path=0.1,
          drop_units=0.1,
          drop_att=0.0,
          pos_encoding='sin_cos',
          dec_proj_mode='mlp',
          enc_drop=0.,
          enc_fuse='pyramid_merge',
          enc_fuse_upsample='nearest',
          enc_fuse_dim=256,
          frozen_backbone=False,
      ),

      optimization=D(
          optimizer='adamw',
          learning_rate=1e-3,
          end_lr_factor=0.01,
          warmup_epochs=10,
          warmup_steps=0,                   # set to >0 to override warmup_epochs.
          weight_decay=0.05,
          global_clipnorm=-1.,
          beta1=0.9,
          beta2=0.95,
          eps=1e-8,
          ema_decay=0.9999,
          ema_name_exact_match=True,
          learning_rate_schedule='linear',
          learning_rate_scaling='none',
      ),

      train=D(
          batch_size=32,
          epochs=40,
          steps=0,                          # set to >0 to override epochs.
          checkpoint_epochs=1,
          checkpoint_steps=0,               # set to >0 to override checkpoint_epochs.
          keep_checkpoint_max=10,
          loss_type='xent',
      ),

      eval=D(
          tag='eval',
          checkpoint_dir='',                # checkpoint_dir will be model_dir if not set.
          batch_size=4,                     # needs to be divisible by total eval examples.
          steps=0,                          # 0 means eval over full validation set.
      ),
  )

  # Update model with architecture variant.
  for key, value in architecture_config_map[encoder_variant].items():
    config.model[key] = value

  # Update decoder architecture variant.
  for key, value in decoder_config_map[decoder_variant].items():
    config.decoder[key] = value

  # on-the-fly gt gathering for metric computation
  # config.dataset.coco_annotations_dir_for_metrics = ''

  config.task.train_transforms = transform_configs.get_panoptic_segmentation_train_transforms(
      image_size, msize, 1.0, 1.0, 0.)
  config.task.eval_transforms = transform_configs.get_panoptic_segmentation_eval_transforms(
      image_size)

  return config
