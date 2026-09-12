# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import numpy as np


class NPNormalGenerator:
    def __init__(
        self,
        mean_low=-100,
        mean_hi=100,
        std_low=1,
        std_high=25,
        output_dtype=np.float16,
    ):
        self.mean_low = mean_low
        self.mean_hi = mean_hi
        self.std_low = std_low
        self.std_hi = std_high
        self.output_dtype = output_dtype

    def generate(self, shape):
        mean = np.random.uniform(low=self.mean_low, high=self.mean_hi)
        std_dev = np.random.uniform(low=self.std_low, high=self.std_hi)

        return np.random.normal(loc=mean, scale=std_dev, size=shape).astype(
            self.output_dtype
        )
