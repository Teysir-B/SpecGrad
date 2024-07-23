# Adapted from https://github.com/lmnt-com/wavegrad under the Apache-2.0 license.

# Copyright 2020 LMNT, Inc. All Rights Reserved.
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

import numpy as np


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is None:
      pass
    else:
      raise NotImplementedError
    return self


params = AttrDict(
    # Training params
    batch_size=32,
    learning_rate=2e-4,
    max_grad_norm=1.0,

    # Data params
    sample_rate=22050,
    hop_samples=300,  # Don't change this. Really.
    n_mels=128,
    fmin=20,
    fmax=11025,
    crop_mel_frames=66,
    std_min=0.01,

    lifter=24,

    # Model params
    noise_schedule=np.linspace(1e-6, 0.01, 1000).tolist(),
    # noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(), ## Training noise schedule from PriorGrad (used in experiments)
    # inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5], ## PG-6
    # inference_noise_schedule=[3e-4, 6e-2, 9e-1], ## WG-3
    inference_noise_schedule=[7e-6, 1.4e-4, 2.1e-3, 2.8e-2, 3.5e-1, 7e-1], ## WG-6
    # inference_noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(), ## WG-50
)
