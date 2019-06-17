import tensorflow as tf
from deepbelief.layers.data import Data
from deepbelief.layers import gplvm
from deepbelief.plotting import LatentPointPlotter
from deepbelief.plotting import LatentSamplePlotter
from deepbelief.util import init_logging

# Import local flags.py module without specifying absolute path
import importlib.util
module_spec = importlib.util.spec_from_file_location('flags', 'flags.py')
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
flags = module.flags

# Note, this script will train the model and create an output folder containing
# checkpoint files and a summary for TensorBoard.

init_logging(flags['training_log_file'])

lp_plotter = LatentPointPlotter(
    xlim=flags['xlim'],
    ylim=flags['ylim'],
    delta=flags['delta_lp'],
    output_dir=flags['latent_points_dir'],
    fig_ext=flags['lp_plotter_fig_ext'])

ls_plotter = LatentSamplePlotter(
    image_shape=flags['img_shape'],
    xlim=flags['xlim'],
    ylim=flags['ylim'],
    delta=flags['delta_ls'],
    num_samples_to_average=flags['num_samples_to_average'],
    output_dir=flags['latent_samples_dir'])

training_data = Data(flags['training_data'],
                     shuffle_first=flags['shuffle_data'],
                     batch_size=flags['training_batch_size'],
                     log_epochs=flags['data_log_epochs'],
                     name='TrainingData')

Y = training_data.next_batch()

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)

kern = gplvm.SEKernel(session=session,
                      alpha=flags['kernel_alpha'],
                      gamma=flags['kernel_gamma'],
                      ARD=flags['kernel_ard'],
                      Q=flags['q'])

layer = gplvm.GPLVM(Y=Y,
                    Q=flags['q'],
                    kern=kern,
                    noise_variance=flags['noise_variance'],
                    latent_point_plotter=lp_plotter,
                    latent_sample_plotter=ls_plotter,
                    session=session,
                    name=flags['gplvm_name'])

layer.build_model()

optimizer = tf.train.AdamOptimizer(learning_rate=flags['learning_rate'])

summary_writer = tf.summary.FileWriter(flags['summary_writer_file'], session.graph)

layer.optimize(optimizer=optimizer,
               num_iterations=flags['num_iterations'],
               eval_interval=flags['eval_interval'],
               ckpt_interval=flags['ckpt_interval'],
               ckpt_dir=flags['ckpt_dir'],
               summary_writer=summary_writer)

session.close()
