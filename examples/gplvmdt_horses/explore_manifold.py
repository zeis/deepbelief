import numpy as np
from scipy import ndimage
import tensorflow as tf
from deepbelief.layers.data import Data
from deepbelief.layers import gplvm
from deepbelief.plotting import LatentSpaceExplorer2D
from deepbelief.util import init_logging

# Import local flags.py module without specifying absolute path
import importlib.util
module_spec = importlib.util.spec_from_file_location('flags', 'flags.py')
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
flags = module.flags

# Note, this script should be run after the model is trained (run train.py first).
# This script will launch two windows, one for the latent space and the other one to
# visualise the ouptut of the model as the pointer moves onto the latent space. If the
# two figures are merged into a single window, separate them first. To activate
# the latent space figure and see the changing output in the other figure it might be
# required to click on the title bar of the window containing the latent space figure.

init_logging(flags['test_log_file'])

assert flags['q'] == 2

latent_space_explorer = LatentSpaceExplorer2D(
    xlim=flags['xlim'],
    ylim=flags['ylim'],
    delta=flags['delta_lp'],
    output_dir=flags['test_plots_dir'],
    sample_shape=flags['img_shape'],
    num_samples_to_average=flags['num_samples_to_average'])

training_data = Data(flags['training_data'],
                     shuffle_first=flags['shuffle_data'],
                     batch_size=flags['training_batch_size'],
                     log_epochs=flags['data_log_epochs'],
                     name='TrainingData')

Y = training_data.next_batch()

# Apply distance transform to the data before passing it to the model
datapoint_length = Y.shape[1]
Y = np.reshape(Y, newshape=(Y.shape[0], flags['img_shape'][0], flags['img_shape'][1]))
Y_copy = np.ndarray.astype(np.copy(Y), dtype=bool)
for ii in range(Y.shape[0]):
    Y[ii, :, :] = ndimage.distance_transform_edt(Y[ii, :, :])
    Y[ii, :, :] -= ndimage.distance_transform_edt(np.invert(Y_copy[ii, :, :]))
Y = np.reshape(Y, newshape=(Y.shape[0], datapoint_length))

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)

kern = gplvm.SEKernel(session=session,
                      alpha=flags['kernel_alpha'],
                      gamma=flags['kernel_gamma'],
                      ARD=flags['kernel_ard'],
                      Q=flags['q'])
kern.restore(flags['kernel_ckpt'])

layer = gplvm.GPLVM(Y=Y,
                    Q=flags['q'],
                    kern=kern,
                    noise_variance=flags['noise_variance'],
                    latent_space_explorer=latent_space_explorer,
                    session=session,
                    name=flags['gplvm_name'])
layer.restore(flags['gplvm_ckpt'])

layer.build_model()
layer.pred_mean = layer.build_pred_mean_thresholded(layer.pred_mean)

layer.explore_2D_latent_space()

session.close()
