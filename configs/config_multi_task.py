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
"""An example config."""

import copy
# pylint: disable=invalid-name,line-too-long

from configs import dataset_configs
from configs.config_base import architecture_config_map
from configs.config_base import D


def get_config(config_str=None):
  """config_str is either empty or contains task,architecture variants."""

  if config_str:
    task_variant, encoder_variant = config_str.split(',')
  else:
    # set default meta-hyperparameters when config_str is not given.
    task_variant = 'object_detection@coco/2017_object_detection+instance_segmentation@2017_instance_segmentation'
    encoder_variant = 'resnet'

  tasks_and_datasets = []
  for task_and_ds in task_variant.split('+'):
    tasks_and_datasets.append(task_and_ds.split('@'))

  image_size = (640, 640)  # Set single image size across the model and tasks

  task_config_map = {
      'object_detection':
          D(
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
              # If specified, outputs including boxes, scores etc. are saved to
              # this file.
              # Note: If any of '{eval_split}', '{top_p}',
              # '{max_instances_per_image_test}', '{step}' are present in the path,
              # they are replaced by the respective values using str.format().
              eval_outputs_json_path='',
              metric=D(name='coco_object_detection',)),
      'instance_segmentation':
          D(
              name='instance_segmentation',
              vocab_id=11,
              object_detection_vocab_id=10,
              image_size=image_size,
              quantization_bins=1000,
              max_instances_per_image=10,
              max_instances_per_image_test=10,  # for debug only
              max_points_per_object=128,
              color_jitter_strength=0.,
              jitter_scale_min=0.3,
              jitter_scale_max=1.0,
              shuffle_polygon_start_point=False,
              top_k=0,
              top_p=0.8,
              temperature=1.0,
              use_gt_box_at_test=True,  # for debug only
              eos_token_weight=1.0,
              weight=1.0,
              ensemble_num_samples=8,
              ensemble_threshold=0.5,
              # Min score of a pre-computed bbox to use for inference.
              # Note ground truth boxes have score 1.0
              min_bbox_score=0.0,
              # Note: max_instances_per_image and max_instances_per_image_test must
              # be set to 1 when using unbatching.
              # Note: unbatch=True can be used alone as well with crop_to_bbox=False
              # This can be used for evaluating on all boxes when increasing
              # max instances causes OOM.
              unbatch=False,
              crop_to_bbox=False,
              crop_to_bbox_pad_scale=0.0,
              metric=D(name='coco_instance_segmentation',)),
      'keypoint_detection':
          D(
              name='keypoint_detection',
              vocab_id=12,
              object_detection_vocab_id=10,
              image_size=image_size,
              quantization_bins=1000,
              max_instances_per_image=1,
              # This is the number of boxes generated from the jointly
              # trained detection model.
              max_instances_per_image_test=1,
              max_points_per_object=17,
              min_bbox_score=0.0,
              color_jitter_strength=0.,
              jitter_scale_min=0.3,
              jitter_scale_max=1.0,
              top_k=0,
              top_p=0.1,
              temperature=1.0,
              use_gt_box_at_test=True,  # for debug only
              invisible_token_weight=0.1,
              eos_token_weight=0.0,
              weight=1.,
              unbatch=True,
              crop_to_bbox=True,
              crop_to_bbox_pad_scale=0.5,
              points_score_weight=1.0,
              eval_suppress_invisible_token=True,
              metric=D(name='coco_keypoint_detection',)),
      'captioning':
          D(name='captioning',
            vocab_id=13,
            image_size=image_size,
            max_seq_len=128,
            captions_per_image=5,
            max_instances_per_image=5,
            color_jitter_strength=0.5,
            jitter_scale_min=1.0,
            jitter_scale_max=1.0,
            eos_token_weight=0.1,
            input_seq_drop_rate=0.5,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            weight=1.,
            metric=D(name='coco_captioning',)),
      'panoptic_segmentation':
          D(
              name='panoptic_segmentation',
              vocab_id=16,
              image_size=image_size,
              object_order='random',
              quantization_bins=1000,
              color_jitter_strength=0.,
              jitter_scale_min=0.3,
              jitter_scale_max=1.0,
              top_k=1,
              top_p=1.0,
              temperature=1.0,
              metric=D(name='coco_panoptic_segmentation', results_dir=''),
              use_model_pred=True,
              min_pixels=2,
              max_instances_per_image=100,
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

      model=D(
          name='encoder_ar_decoder',
          image_size=image_size,
          max_seq_len=512,
          vocab_size=35000,                 # Note: should be large enough for 100 + num_classes + quantization_bins + (optional) text
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
          learning_rate=1e-4,
          end_lr_factor=0.01,
          warmup_epochs=10,
          warmup_steps=0,
          weight_decay=0.05,
          global_clipnorm=-1,
          beta1=0.9,
          beta2=0.95,
          eps=1e-8,
          learning_rate_schedule='linear',
          learning_rate_scaling='none',
      ),

      train=D(
          batch_size=128,
          epochs=100,
          steps=0,
          checkpoint_epochs=1,
          checkpoint_steps=0,
          keep_checkpoint_max=10,
          loss_type='xent',
      ),

      eval=D(
          tag='eval',
          checkpoint_dir='',
          # checkpoint_dir=get_multi_task_checkpoint_dir(encoder_variant, image_size),
          batch_size=8,
          steps=0,
      ),

      tokenizer=D(
          sentencepiece_model=get_tokenizer_path(),
          add_bos=False,
          add_eos=True,
      ),
  )

  # Update model with architecture variant.
  for key, value in architecture_config_map[encoder_variant].items():
    config.model[key] = value

  return config


def get_obj365_pretrained_checkpoint(encoder_variant):  # pylint: disable=missing-function-docstring
  CKPT_PREFIX = 'gs://pix2seq'
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


def get_multi_task_checkpoint_dir(encoder_variant, image_size):  # pylint: disable=missing-function-docstring
  CKPT_PREFIX = 'gs://pix2seq'
  if encoder_variant == 'vit-b' and image_size == (640, 640):
    return f'{CKPT_PREFIX}/multi_task/ckpt/vit_b_640x640'
  elif encoder_variant == 'vit-b' and image_size == (1024, 1024):
    return f'{CKPT_PREFIX}/multi_task/ckpt/vit_b_1024x1024'
  else:
    raise ValueError('Unknown encoder_variant {} or image_size {}'.format(encoder_variant, image_size))


def get_tokenizer_path():  # pylint: disable=missing-function-docstring
  PREFIX = 'gs://pix2seq'
  return f'{PREFIX}/multi_task/data/c4_en_32k_spm.model'
