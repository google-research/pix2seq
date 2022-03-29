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
"""Config file for object detection fine-tuning and evaluation."""

import copy
# pylint: disable=invalid-name,line-too-long,missing-docstring
from configs.config_base import architecture_config_map
from configs.config_base import D


task_specific_coco_dataset_config = {
    'object_detection':
        D(
            name='object_detection',
            tfds_name='coco/2017',
            train_filename='instances_train2017.json',
            val_filename='instances_val2017.json',
        ),
}


def get_config(config_str=None):
  """config_str is either empty or contains task,architecture variants."""

  task_variant = 'object_detection'
  encoder_variant = 'vit-b'                 # Set model architecture.
  image_size = 640                          # Set image size.


  coco_annotations_dir = 'annotations'

  task_config_map = {
      'object_detection': D(
          name='object_detection',
          vocab_id=10,
          image_size=image_size,
          quantization_bins=1000,
          max_instances_per_image=100,
          max_instances_per_image_test=100,
          object_order='random',
          color_jitter_strength=0.,
          jitter_scale_min=0.3,
          jitter_scale_max=2.0,
          # Train on both ground-truth and (augmented) noisy objects.
          noise_bbox_weight=1.0,
          eos_token_weight=0.1,
          # Train on just ground-truth objects (with an ending token).
          # noise_bbox_weight=0.0,
          # eos_token_weight=0.1,
          class_label_corruption='rand_n_fake_cls',
          top_k=0,
          top_p=0.4,
          temperature=1.0,
          weight=1.0,
      ),
  }

  shared_coco_dataset_config = D(
      train_split='train',
      eval_split='validation',
      coco_annotations_dir=coco_annotations_dir,
      batch_duplicates=1,
      cache_dataset=False,
      label_shift=0,
  )

  task_d_list = []
  dataset_list = []
  for tv in task_variant.split('+'):
    task_d_list.append(task_config_map[tv])
    dataset_config = copy.deepcopy(shared_coco_dataset_config)
    dataset_config.update(task_specific_coco_dataset_config[tv])
    dataset_list.append(dataset_config)

  config = D(
      dataset=dataset_list[0],
      datasets=dataset_list,

      task=task_d_list[0],
      tasks=task_d_list,

      model=D(
          name='encoder_ar_decoder',
          image_size=image_size,
          max_seq_len=512,
          vocab_size=3000,                  # Note: should be large enough for 100 + num_classes + quantization_bins + (optional) text
          coord_vocab_shift=1000,           # Note: make sure num_class <= coord_vocab_shift - 100
          text_vocab_shift=3000,            # Note: make sure coord_vocab_shift + quantization_bins <= text_vocab_shift
          use_cls_token=False,
          shared_decoder_embedding=True,
          decoder_output_bias=True,
          patch_size=16,
          drop_path=0.1,
          drop_units=0.1,
          drop_att=0.0,
          dec_proj_mode='mlp',
          pos_encoding='sin_cos',
          pos_encoding_dec='learned',
          pretrained_ckpt=get_obj365_pretrained_checkpoint(encoder_variant),
      ),

      optimization=D(
          optimizer='adamw',
          learning_rate=3e-5,
          end_lr_factor=0.01,
          warmup_epochs=2,
          warmup_steps=0,                   # set to >0 to override warmup_epochs.
          weight_decay=0.05,
          global_clipnorm=-1,
          beta1=0.9,
          beta2=0.95,
          eps=1e-8,
          learning_rate_schedule='linear',
          learning_rate_scaling='none',
      ),

      train=D(
          batch_size=32,
          epochs=40,
          steps=0,                          # set to >0 to override epochs.
          checkpoint_epochs=1,
          checkpoint_steps=0,               # set to >0 to override checkpoint_epochs.
          keep_checkpoint_max=5,
          loss_type='xent',
      ),

      eval=D(
          tag='eval',
          checkpoint_dir='',                # checkpoint_dir will be model_dir if not set.
          # checkpoint_dir=get_coco_finetuned_checkpoint(encoder_variant, image_size),
          batch_size=8,                     # needs to be divisible by total eval examples.
          steps=0,                          # 0 means eval over full validation set.
      ),
  )

  # Update model with architecture variant.
  for key, value in architecture_config_map[encoder_variant].items():
    config.model[key] = value

  return config

CKPT_PREFIX = 'gs://pix2seq'


def get_obj365_pretrained_checkpoint(encoder_variant):
  if encoder_variant == 'resnet':
    return f'{CKPT_PREFIX}/obj365_pretrain/resnet_640x640_b256_s400k'
  elif encoder_variant == 'resnet-c':
    return f'{CKPT_PREFIX}/obj365_pretrain/resnetc_640x640_b256_s400k'
  elif encoder_variant == 'vit-b':
    return f'{CKPT_PREFIX}/obj365_pretrain/vit_b_640x640_b256_s400k'
  elif encoder_variant == 'vit-l':
    return f'{CKPT_PREFIX}/obj365_pretrain/vit_l_640x640_b256_s400k'
  else:
    raise ValueError('Unknown encoder_variant {}'.format(encoder_variant))


def get_coco_finetuned_checkpoint(encoder_variant, image_size):
  if encoder_variant == 'resnet':
    return f'{CKPT_PREFIX}/coco_det_finetune/resnet_{image_size}x{image_size}'
  elif encoder_variant == 'resnet-c':
    return f'{CKPT_PREFIX}/coco_det_finetune/resnetc_{image_size}x{image_size}'
  elif encoder_variant == 'vit-b':
    return f'{CKPT_PREFIX}/coco_det_finetune/vit_b_{image_size}x{image_size}'
  elif encoder_variant == 'vit-l':
    return f'{CKPT_PREFIX}/coco_det_finetune/vit_l_{image_size}x{image_size}'
  else:
    raise ValueError('Unknown encoder_variant {}'.format(encoder_variant))
