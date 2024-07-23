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
import os
import random
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

from glob import glob
from torch.utils.data.distributed import DistributedSampler
from preprocess import get_spec_filter
from utils import istft_M_stft

class NumpyDataset(torch.utils.data.Dataset):
  def __init__(self, paths, files, params, is_training=True):
    super().__init__()
    self.params = params
    self.is_training = is_training
    self.filenames = []
    if not files:
      for path in paths:
        self.filenames += glob(f'{path}/**/*.wav', recursive=True)
    else:
      assert len(files) == len(paths)
      for path, f in zip(paths, files):
        with open(f, 'r', encoding='utf-8') as fi:
          self.filenames += [os.path.join(path, x.split('|')[0] + '.wav')
                              for x in fi.read().split('\n') if len(x) > 0]

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    signal, sr = torchaudio.load(audio_filename)
    signal = torch.nn.functional.normalize(signal, p=float('inf'), dim=-1, eps=1e-12)*0.95
    signal = signal.squeeze(0)
    if self.params.sample_rate != sr:
      raise ValueError(f'Invalid sample rate {sr}.')

    if self.is_training:
      start = random.randint(0, signal.shape[0] - (self.params.crop_mel_frames - 1) * self.params.hop_samples)
      end = start + (self.params.crop_mel_frames - 1) * self.params.hop_samples
      # get segment of audio
      signal = signal[start:end]

    spectrogram, M = get_spec_filter(signal, self.params)
    if self.is_training:
      signal = torch.hstack([signal, torch.zeros(self.params.hop_samples)])
    else:
      signal = torch.hstack([signal, torch.zeros(spectrogram.squeeze(0).shape[-1]*self.params.hop_samples - len(signal))])
      assert len(signal) % self.params.hop_samples == 0
    return {
        'audio': signal,
        'spectrogram': spectrogram.squeeze(0).T,
        'filter': M.squeeze(0)
    }


class Collator:
  def __init__(self, params, is_training = True):
    self.params = params
    self.is_training = is_training

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    for record in minibatch:
      # Filter out records that aren't long enough.
      if len(record['spectrogram']) < self.params.crop_mel_frames:
        del record['spectrogram']
        del record['audio']
        del record['filter']
        continue
      record['spectrogram'] = record['spectrogram'].T
      record['filter'] = record['filter']
      record['audio'] = record['audio']

    audio = torch.stack([record['audio'] for record in minibatch if 'audio' in record])
    spectrogram = torch.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
    filter_ = torch.stack([record['filter'] for record in minibatch if 'filter' in record])
    return {
        'audio': audio,
        'spectrogram': spectrogram,
        'filter': filter_
    }


def from_path(data_dirs, training_files, params, is_distributed=False):
  dataset = NumpyDataset(data_dirs, training_files, params, is_training=True)
  print(len(dataset), "files for training")
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate,
      shuffle=not is_distributed,
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True,
      num_workers=os.cpu_count())

def from_path_valid(data_dirs, validation_files, params, is_distributed=False):
  dataset = NumpyDataset(data_dirs, validation_files, params, is_training=False)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=1,
      collate_fn=Collator(params, is_training = False).collate,
      shuffle=False,
      num_workers=1,
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=False,
      drop_last=False)
