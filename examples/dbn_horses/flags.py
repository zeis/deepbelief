import os
import sys

flags = {}

# Note, this script is not intended to be run. It just contains flags for the other scripts.

###############################################################################
# Data
###############################################################################
flags['training_data'] = os.path.join('..', '_datasets', 'weizmann_horses_training_328x1024_binary.h5')
flags['test_data'] = os.path.join('..', '_datasets', 'weizmann_horses_eslami_test_14x1024_binary.h5')

flags['img_shape'] = (32, 32)
flags['shuffle_data'] = False
flags['data_log_epochs'] = False

###############################################################################
# Model
###############################################################################
flags['layer_1_num_h'] = 200
flags['layer_1_W_std'] = 0.01
flags['layer_1_name'] = 'BBRBM-' + str(flags['layer_1_num_h'])

flags['layer_2_num_h'] = 100
flags['layer_2_W_std'] = 0.01
flags['layer_2_name'] = 'BBRBM-' + str(flags['layer_1_num_h']) + '-' + str(flags['layer_2_num_h'])

flags['layer_3_num_h'] = 50
flags['layer_3_W_std'] = 0.01
flags['layer_3_name'] = 'BBRBM-' + str(flags['layer_2_num_h']) + '-' + str(flags['layer_3_num_h'])


flags['training_batch_size'] = 328
flags['test_batch_size'] = 8
flags['learning_rate'] = 0.001
flags['pcd'] = False
flags['num_gibbs_steps'] = 10
flags['num_iterations'] = 20000
flags['eval_interval'] = 1000
flags['ckpt_interval'] = flags['num_iterations']

# Number of samples to average for each latent point to generate a  greyscale image.
flags['num_samples_to_average'] = 100

###############################################################################
# Generalisation Test
###############################################################################
flags['test_generalisation_batch_size'] = 14
flags['test_generalisation_num_iterations'] = 1
flags['test_generalisation_num_runs'] = 1
flags['test_generalisation_num_samples_to_average'] = 100

###############################################################################
# Plots
###############################################################################
flags['plot_y_lim'] = 600

###############################################################################
# Output Folders
###############################################################################
flags['output_dir'] = 'output'
flags['ckpt_dir'] = os.path.join(flags['output_dir'], 'checkpoints')
flags['test_loss_plots_dir'] = os.path.join(flags['output_dir'], 'test_loss_plots')
flags['training_figures_dir'] = os.path.join(flags['output_dir'], 'training_figures')
flags['test_plots_dir'] = os.path.join(flags['output_dir'], 'test_plots')
flags['test_generalisation_plots_dir'] = os.path.join(flags['test_plots_dir'], 'generalisation')

flags['training_log_file'] = os.path.join(flags['output_dir'], 'training.log')
flags['test_log_file'] = os.path.join(flags['output_dir'], 'test.log')

flags['layer_1_ckpt'] = os.path.join(
    flags['ckpt_dir'], flags['layer_1_name'] + '-' + str(flags['num_iterations']))
flags['layer_2_ckpt'] = os.path.join(
    flags['ckpt_dir'], flags['layer_2_name'] + '-' + str(flags['num_iterations']))
flags['layer_3_ckpt'] = os.path.join(
    flags['ckpt_dir'], flags['layer_3_name'] + '-' + str(flags['num_iterations']))

os.makedirs(flags['output_dir'], exist_ok=True)
os.makedirs(flags['ckpt_dir'], exist_ok=True)
os.makedirs(flags['test_loss_plots_dir'], exist_ok=True)
os.makedirs(flags['training_figures_dir'], exist_ok=True)
os.makedirs(flags['test_plots_dir'], exist_ok=True)
os.makedirs(flags['test_generalisation_plots_dir'], exist_ok=True)
