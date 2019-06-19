**Status:** Archive (code is provided as-is, no updates expected).

# deepbelief

This library contains code to define and train a GPDBN model (Gaussian Process Deep Belief Network: https://arxiv.org/abs/1812.05477) as well as other models including deep belief networks and GPLVMs.

## Installation

Requirements: Python 3.5 and TensorFlow 1.8 (see setup.py for a full list of requirements).

To install TensorFlow refer to: https://www.tensorflow.org/install/

Install `deepbelief` via pip:

    cd <deepbelief_folder>
    pip install -e .

The above commands will install `deepbelief` in editable mode, in this way pip will make links pointing to the source code (so that the code can be modified and tested easily).

## Experiment Examples

Some model examples with related experiments can be found in the folder `examples`.

Some example datasets are contained in the subfolder `_datasets`. These are binary files in HDF5 format:

- `weizmann_horses_training_328x1024_binary.h5`: 328 32x32 training binary images from the [Weizmann horses dataset](http://www.msri.org/people/members/eranb/).
- `weizmann_horses_eslami_test_14x1024_binary.h5`: 14 32x32 test binary images of horses from the Shape Boltzmann Machine paper of [Eslami et al](http://arkitus.com/files/cvpr-12-eslami-sbm.pdf).
- `mnist_training_5000x728_equal_classes_binary.h5`: 5000 28x28 training binary imaged from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
- `mnist_test_30x728_equal_classes_binary.h5`: 30 28x28 test binary imaged from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Mini-batched GPDBN

The `mini-batch` branch contains a version of the GPDBN model that uses mini-batching at training and test time. This was developed in collaboration with Erik Bodin ([@bodin-e](https://github.com/bodin-e)). 

## Unit Tests

    python -m unittest -v
