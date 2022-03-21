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
"""General Registry."""

from typing import Any
from typing import Callable


class Registry(object):
  """Registry."""

  def __init__(self):
    self._registry = {}

  def register(self, key: str) -> Callable[[Any], None]:
    """Returns callable to register value for key."""
    def r(item):
      if key in self._registry:
        raise ValueError("%s already registered!" % key)
      self._registry[key] = item
      return item
    return r

  def lookup(self, key: str) -> Any:
    """Looks up value for key."""
    if key not in self._registry:
      valid_keys = "\n".join(self._registry.keys())
      raise ValueError(
          "%s not registered!\n\n"
          "Valid keys:%s\n\n" %
          (key, valid_keys))
    return self._registry[key]
