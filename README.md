# Implementation of SpecGrad (Unofficial)

This repository contains an unofficial implementation of the paper [SpecGrad: Diffusion Probabilistic Model based Neural Vocoder with Adaptive Noise Spectral Shaping](https://arxiv.org/abs/2203.16749).

## Training

To train the model, run the `__main__.py` script with various arguments to customize the training process. Each argument is briefly described within the file for your reference.

## Inference
The `inference.py` script enables you to perform inference using a trained model. To utilize this script, run the following command:<br/>

  ```
  python inference.py --model_dir ./model --spectrogram_path ./spectrograms/example_spec.npy --output result.wav
  ```

## References
- [SpecGrad: Diffusion Probabilistic Model based Neural Vocoder with Adaptive Noise Spectral Shaping](https://arxiv.org/abs/2203.16749)
- [WaveGrad](https://github.com/lmnt-com/wavegrad)

## Contribution

We welcome contributions to improve the organization and computational efficiency of this code. Please feel free to contribute or report any issues or bugs.

## Citation

If this implementation is helpful in your research, please consider citing the paper [GLA-Grad: A Griffin-Lim Extended Waveform Generation Diffusion Model](https://arxiv.org/abs/2402.15516), for which this implementation was used, with the following BibTeX entry:

```
@inproceedings{liu2024glagrad,
  title={GLA-Grad: A Griffin-Lim Extended Waveform Generation Diffusion Model},
  author={Liu, Haocheng and Baoueb, Teysir and Fontaine, Mathieu and Le Roux, Jonathan and Richard, Gaël},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing, Apr 2024, Seoul (Korea), South Korea},
  year={2024}
}
```
