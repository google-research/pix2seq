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
"""Vocab."""

# A shared vocab among tasks and its structure -
# Special tokens: [0, 99).
# Class tokens: [100, coord_vocab_shift). Total coord_vocab_shift - 100 classes.
# Coordinate tokens: [coord_vocab_shift, text_vocab_shift).
# Text tokens: [text_vocab_shift, ...].

PADDING_TOKEN = 0

# 10-29 reserved for task id.

FAKE_CLASS_TOKEN = 30
FAKE_TEXT_TOKEN = 30  # Same token to represent fake class and fake text.
SEPARATOR_TOKEN = 40
INVISIBLE_TOKEN = 41

BASE_VOCAB_SHIFT = 100

# Floats used to represent padding and separator in the flat list of polygon
# coords, and invisibility in the key points.
PADDING_FLOAT = -1.
SEPARATOR_FLOAT = -2.
INVISIBLE_FLOAT = -3.
FLOATS = [PADDING_FLOAT, SEPARATOR_FLOAT, INVISIBLE_FLOAT]
TOKENS = [PADDING_TOKEN, SEPARATOR_TOKEN, INVISIBLE_TOKEN]
FLOAT_TO_TOKEN = dict(zip(FLOATS, TOKENS))
TOKEN_TO_FLOAT = dict(zip(TOKENS, FLOATS))
