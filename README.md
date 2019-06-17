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

Two example datasets are contained in the subfolder `_datasets`. These are two binary files in HDF5 format. The training dataset contains 328 32x32 binary images from the [Weizmann horses dataset](http://www.msri.org/people/members/eranb/). The test dataset contains 14 32x32 binary images of horses from the Shape Boltzmann Machine paper of [Eslami et al](http://arkitus.com/files/cvpr-12-eslami-sbm.pdf).

## Unit Tests

    python -m unittest -v
