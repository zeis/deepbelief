import tensorflow as tf
from deepbelief.layers.rbm import RBM
from deepbelief.layers.gprbm import GPRBM
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

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)

layer1 = RBM(session=session,
             num_v=flags['img_shape'][0] * flags['img_shape'][1],
             num_h=flags['layer_1_num_h'],
             temperature=flags['temperature'],
             name=flags['layer_1_name'])

layer1.restore(flags['layer_1_ckpt'])

layer2 = RBM(session=session,
             num_v=flags['layer_1_num_h'],
             num_h=flags['layer_2_num_h'],
             temperature=flags['temperature'],
             bottom=layer1,
             name=flags['layer_2_name'])

layer2.restore(flags['layer_2_ckpt'])

kern = gplvm.SEKernel(session=session,
                      alpha=flags['kernel_alpha'],
                      gamma=flags['kernel_gamma'],
                      ARD=flags['kernel_ard'],
                      Q=flags['q'])
kern.restore(flags['kernel_ckpt'])

layer3 = GPRBM(num_v=flags['layer_2_num_h'],
               num_h=flags['layer_3_num_h'],
               Q=flags['q'],
               N=flags['training_batch_size'],
               eval_flag=True,
               kern=kern,
               noise_variance=flags['noise_variance'],
               fixed_noise_variance=flags['fixed_noise_variance'],
               temperature=flags['temperature'],
               bottom=layer2,
               latent_space_explorer=latent_space_explorer,
               session=session,
               name=flags['layer_3_name'])

layer3.restore(flags['layer_3_ckpt'])

layer3.build_model()

layer3.explore_2D_latent_space()

session.close()
