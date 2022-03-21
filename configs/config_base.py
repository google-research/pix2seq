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
"""Common / shared settings among multiple configs."""

import ml_collections


def D(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


architecture_config_map = {
    'vit-b': D(
        resnet_variant='c1',
        num_encoder_layers=12,
        dim_att=768,
        dim_mlp=3072,
        num_heads=12,
        num_decoder_layers=6,
        dim_att_dec=512,
        dim_mlp_dec=2048,
        num_heads_dec=16,
    ),
    'vit-l': D(
        resnet_variant='c1',
        num_encoder_layers=24,
        dim_att=1024,
        dim_mlp=4096,
        num_heads=16,
        num_decoder_layers=8,
        dim_att_dec=512,
        dim_mlp_dec=2048,
        num_heads_dec=16,
    ),
    'resnet': D(
        resnet_variant='standard',
        resnet_depth=50,
        resnet_sk_ratio=0.,
        resnet_width_multiplier=1,
        num_encoder_layers=6,
        dim_att=256,
        dim_mlp=1024,
        num_heads=8,
        num_decoder_layers=6,
        dim_att_dec=256,
        dim_mlp_dec=1024,
        num_heads_dec=8,
    ),
    'resnet-c': D(
        resnet_variant='c4',
        resnet_depth=50,
        resnet_sk_ratio=0.,
        resnet_width_multiplier=1,
        num_encoder_layers=12,
        dim_att=512,
        dim_mlp=2048,
        num_heads=16,
        num_decoder_layers=8,
        dim_att_dec=512,
        dim_mlp_dec=2048,
        num_heads_dec=16,
    ),
}
