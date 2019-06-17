import tensorflow as tf
from deepbelief.layers.data import Data
from deepbelief.layers.rbm import RBM
from deepbelief.layers.sbm_lower import SBM_Lower
from deepbelief.plotting import GraphPlotter
from deepbelief.plotting import ImageRowPlotter
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

loss_plotter = GraphPlotter(xlim=(0, flags['num_iterations']),
                            ylim=(0, flags['plot_y_lim']),
                            xlabel='Iteration',
                            ylabel='Loss',
                            output_dir=flags['test_loss_plots_dir'])

# All the layers can share a single distr_potter
distr_plotter = ImageRowPlotter(num_images=2,
                                image_shape=flags['img_shape'],
                                output_dir=flags['training_figures_dir'])

training_data = Data(flags['training_data'],
                     shuffle_first=flags['shuffle_data'],
                     batch_size=flags['training_batch_size'],
                     log_epochs=flags['data_log_epochs'],
                     name='TrainingData')
test_data = Data(flags['test_data'],
                 shuffle_first=flags['shuffle_data'],
                 batch_size=flags['test_batch_size'],
                 log_epochs=flags['data_log_epochs'],
                 name='TestData')

optimizer = tf.train.AdamOptimizer(learning_rate=flags['learning_rate'])

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)

layer1 = SBM_Lower(session=session,
                   side=flags['img_shape'][0],
                   side_overlap=flags['layer_1_side_overlap'],
                   num_h=flags['layer_1_num_h'],
                   lr=flags['learning_rate'],
                   W_std=flags['layer_1_W_std'],
                   loss_plotter=loss_plotter,
                   distr_plotter=distr_plotter,
                   name=flags['layer_1_name'])

layer1.train(optimizer=optimizer,
             training_data=training_data,
             test_data=test_data,
             num_gibbs_steps=flags['num_gibbs_steps'],
             pcd=flags['pcd'],
             max_iterations=flags['num_iterations'],
             eval_interval=flags['eval_interval'],
             ckpt_interval=flags['ckpt_interval'],
             ckpt_dir=flags['ckpt_dir'])

layer2 = RBM(session=session,
             num_v=flags['layer_1_num_h'],
             num_h=flags['layer_2_num_h'],
             lr=flags['learning_rate'],
             W_std=flags['layer_1_W_std'],
             bottom=layer1,
             loss_plotter=loss_plotter,
             distr_plotter=distr_plotter,
             name=flags['layer_2_name'])

layer2.train(optimizer=optimizer,
             training_data=training_data,
             test_data=test_data,
             num_gibbs_steps=flags['num_gibbs_steps'],
             pcd=flags['pcd'],
             max_iterations=flags['num_iterations'],
             eval_interval=flags['eval_interval'],
             ckpt_interval=flags['ckpt_interval'],
             ckpt_dir=flags['ckpt_dir'])

session.close()
