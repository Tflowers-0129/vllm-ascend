#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import inspect
from functools import wraps

try:
    import vllm.model_executor.models.gemma4_mm as gemma4_mm
except ImportError:
    gemma4_mm = None


if gemma4_mm is not None:
    _ORIG_GEMMA4_MM_INIT = gemma4_mm.Gemma4ForConditionalGeneration.__init__
    _ORIG_AUTO_MODEL_FROM_CONFIG = inspect.getattr_static(
        gemma4_mm.AutoModel, "from_config"
    )

    def _call_orig_from_config(*args, **kwargs):
        return _ORIG_AUTO_MODEL_FROM_CONFIG.__get__(
            None, gemma4_mm.AutoModel
        )(*args, **kwargs)


    @wraps(_ORIG_GEMMA4_MM_INIT)
    def _patched_gemma4_mm_init(self, *, vllm_config, prefix=""):
        @wraps(gemma4_mm.AutoModel.from_config)
        def _from_config(cls, *args, **kwargs):
            kwargs.setdefault("torch_dtype", vllm_config.model_config.dtype)
            return _call_orig_from_config(*args, **kwargs)

        gemma4_mm.AutoModel.from_config = classmethod(_from_config)
        try:
            return _ORIG_GEMMA4_MM_INIT(
                self,
                vllm_config=vllm_config,
                prefix=prefix,
            )
        finally:
            gemma4_mm.AutoModel.from_config = _ORIG_AUTO_MODEL_FROM_CONFIG


    gemma4_mm.Gemma4ForConditionalGeneration.__init__ = _patched_gemma4_mm_init
