import os
import datetime

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
flags['layer_3_W_std'] = 0.0001
flags['layer_3_name'] = 'GPRBM-' + str(flags['layer_2_num_h']) + '-' + str(flags['layer_3_num_h'])

flags['kernel_name'] = 'SEKernel'
# flags['kernel_alpha'] = 1.0
flags['kernel_alpha'] = False
flags['kernel_gamma'] = False
flags['kernel_ard'] = False

flags['q'] = 2

flags['training_batch_size'] = 328
flags['learning_rate'] = 0.001
flags['noise_variance'] = 0.01
flags['fixed_noise_variance'] = True
flags['temperature'] = 0.1
flags['num_iterations'] = 5000
flags['fixed_X_num_iterations'] = 1000
flags['eval_interval'] = 1000
flags['ckpt_interval'] = flags['num_iterations']

# Number of samples to average for each latent point to generate a  greyscale image.
flags['num_samples_to_average'] = 100

###############################################################################
# Generalisation Test
###############################################################################
flags['test_generalisation_batch_size'] = 14
flags['test_generalisation_learning_rate'] = 0.01
flags['test_generalisation_log_var_scaling'] = 0.1
flags['test_generalisation_num_iterations'] = 50
flags['test_generalisation_num_runs'] = flags['training_batch_size']
flags['test_generalisation_num_samples_to_average'] = 100

###############################################################################
# Plots
###############################################################################
flags['xlim'] = flags['ylim'] = (-3, 3)  # Latent space ranges
flags['delta_ls'] = 0.3  # Latent image sample grid step size
flags['delta_lp'] = 0.01  # Latent plot grid step size
flags['lp_plotter_fig_ext'] = '.png'

###############################################################################
# Output Folders
###############################################################################
flags['output_dir'] = 'output'
flags['ckpt_dir'] = os.path.join(flags['output_dir'], 'checkpoints')
flags['latent_points_dir'] = os.path.join(flags['output_dir'], 'latent_points')
flags['latent_samples_dir'] = os.path.join(flags['output_dir'], 'latent_samples')
flags['test_plots_dir'] = os.path.join(flags['output_dir'], 'test_plots')
flags['test_generalisation_plots_dir'] = os.path.join(flags['test_plots_dir'], 'generalisation')

flags['training_log_file'] = os.path.join(flags['output_dir'], 'training.log')
flags['test_log_file'] = os.path.join(flags['output_dir'], 'test.log')
flags['summary_writer_file'] = os.path.join(
    flags['output_dir'], 'summary' + datetime.datetime.now().strftime('_%Y.%m.%d_%H.%M.%S'))

flags['kernel_ckpt'] = os.path.join(
    flags['ckpt_dir'], flags['kernel_name'] + '-' + str(flags['num_iterations']))
flags['layer_1_ckpt'] = os.path.join(
    flags['ckpt_dir'], flags['layer_1_name'] + '-' + str(flags['num_iterations']))
flags['layer_2_ckpt'] = os.path.join(
    flags['ckpt_dir'], flags['layer_2_name'] + '-' + str(flags['num_iterations']))
flags['layer_3_ckpt'] = os.path.join(
    flags['ckpt_dir'], flags['layer_3_name'] + '-' + str(flags['num_iterations']))

os.makedirs(flags['output_dir'], exist_ok=True)
os.makedirs(flags['ckpt_dir'], exist_ok=True)
os.makedirs(flags['latent_points_dir'], exist_ok=True)
os.makedirs(flags['latent_samples_dir'], exist_ok=True)
os.makedirs(flags['test_plots_dir'], exist_ok=True)
os.makedirs(flags['test_generalisation_plots_dir'], exist_ok=True)
