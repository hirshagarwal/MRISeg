# MRISeg
Re-trainable MRI image segmentation model with Metal Performance Shader acceleration.

## Training Data
The data used for training the initial model is the BraTS2020 data found here:
[BraTS2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation?rvi=1)

## Metal Performance Shaders
This system was designed to be accelerated using Apple silicon
and the metal performance shaders. The default device can be changed to cpu.

### Benchmarks
Initial benchmarking shows approximately a 6x speedup using MPS on
M2 Pro with 32GB ram.
