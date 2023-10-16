# MRISeg

Re-trainable MRI brain image segmentation model with Metal Performance Shader acceleration.
Model is based on original [SegNet Paper](https://arxiv.org/pdf/1511.00561.pdf) which in turn
is based on the VGG16 model.

## Abstract
While machine learning models exist to do image segmentation of medical imaging they generally
require significant bespoke data to train. This model is designed to segment brain images and 
be transferred to other similar tasks using consumer hardware acceleration. While the principle
behind this type of transfer learning has been explored, trying to generalize a model that has
been trained on a specific data type (e.g. MRI Brain) and then generalizing it to other tasks
in a similar domain seems like a more efficient approach. Hopefully this model will be able to 
create performant image segmentation for various radiology or pathology based tasks in the future.

## Training Data
The data used for training the initial model is the BraTS2020 data found here:
[BraTS2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation?rvi=1)

This dataset consists of 369 volumes, each with 5 channels and 154 depth layers. The channels
available are:
1. T1
2. T1CE
3. T2
4. FLAIR
5. Segmentation

The segmentation layer is the desired output layer, and it is simply an image of identical size
that outputs a mask corresponding to the tumor found in its associated volume.

## Metal Performance Shaders
This system was designed to be accelerated using Apple silicon
and the metal performance shaders. The default device can be changed to cpu.

There are still some limitations with this system. Right now one limitation is the `max_unpool2d`
operation is not available for MPS. Writing this operating into PyTorch might at some point be
desirable, however for now the CPU fallback is sufficient.


### Benchmarks
Initial benchmarking shows approximately a 6x speedup using MPS on
M2 Pro with 32GB ram. On this machine a batch of 10 images will run in about 0.6 seconds.
This speed is generally adequate for training the model, and will likely be more than
sufficient for transfer learning. With more data the training should run on CUDA with
equal or superior performance, so this might be a reasonable avenue to accelerate training.

## Base Segmentation
The base segmentation model is designed to segment MRI images from the BraTS2020 dataset.
This primarily consists of Gliomas. The training uses T1, T2 and FLAIR sequences as input and a
a binary segmentation mask as the target. The original dataset includes T1CE images, however
in its current for the model doesn't utilize them for training. This might be a reasonable inclusion,
however more investigation is necessary to ascertain whether there is enough of an information advantage.

### Example Training Segmentation
At baseline random weights produces a segmentation pattern that is essentially noise.
An example of the untrained net can be found below.
![Example Untrained Image](images/image_0_0.png)

After approximately one epoch the model begins to classify and segment images accurately, however
with poor spatial resolution. Many of the samples contain no visible tumor, and hence should output blank
segmentation maps. This combined with the relatively small tumor size creates a strong negative output 
weighting (i.e. a model outputting 100% blank images will have a relatively high accuracy). In order to
combat this a positive output weighting is applied the loss algorithm.
![Example Trained Image](images/image_1_200.png)