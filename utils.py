import torch
import torchaudio
import numpy as np
from params import params
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt


def ConcatNegativeFrequency(tensor):
  return torch.concat((tensor[..., :-1], tensor[..., 1:].flip(dims = (-1,))), -1)


def MinimumPhaseFilter(amplitude):
  """Computes minimum phase filter from input amplitude.

  Args:
    amplitude: amplitude spectra, shape=[..., num_bins]

  Returns:
    Minimum phase frequency response, dtype=tf.complex64, the same shape.
  """
  rank = amplitude.ndim
  num_bins = amplitude.shape[-1]
  amplitude = ConcatNegativeFrequency(amplitude)

  fftsize = (num_bins - 1) * 2
  m0 = torch.zeros((fftsize // 2 - 1,), dtype=torch.complex64)
  m1 = torch.ones((1,), dtype=torch.complex64)
  m2 = torch.ones((fftsize // 2 - 1,), dtype=torch.complex64) * 2.0
  minimum_phase_window = torch.concat([m1, m2, m1, m0], axis=0)

  if rank > 1:
    new_shape = [1] * (rank - 1) + [fftsize]
    minimum_phase_window = torch.reshape(minimum_phase_window, new_shape)

  cepstrum = torch.fft.ifft(torch.log(amplitude).to(torch.complex64))
  windowed_cepstrum = cepstrum * minimum_phase_window
  imag_phase = torch.imag(torch.fft.fft(windowed_cepstrum))
  phase = torch.exp(torch.complex(imag_phase * 0.0, imag_phase))
  minimum_phase = amplitude.to(torch.complex64) * phase
  return minimum_phase[..., :num_bins]


def istft_M_stft(audio, M, is_valid = False):
  hop = params.hop_samples
  win = hop * 4
  n_fft = 2**((win-1).bit_length())
  ## to compute stft
  Spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win, hop_length=hop, power=None, normalized=False, pad_mode='reflect',
                                           center = True, onesided=True).to(audio.device)
  ## to compute istft
  iSpec = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, win_length=win, hop_length=hop, normalized=False, pad_mode='reflect',
                                                   center=True, onesided=True).to(audio.device)

  audio_len = iSpec(M).shape[-1]
  audio_new = M * Spec(audio[..., :audio_len])
  audio_new = iSpec(audio_new, audio.shape[-1])
  if is_valid:
    return audio_new, audio_len
  return audio_new

# from https://github.com/jik876/hifi-gan/blob/master/utils.py
def plot_spectrogram(spectrogram):
  fig, ax = plt.subplots(figsize=(10, 2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)

  fig.canvas.draw()
  plt.close()

  return fig

def plot_audio(audio, sr):
  fig = plt.figure(figsize=(10, 2))
  plt.plot(np.arange(len(audio)) / sr, audio)
  return fig
