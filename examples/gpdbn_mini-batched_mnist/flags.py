import os
import datetime

flags = {}

###############################################################################
# Data
###############################################################################
flags['training_data'] = os.path.join('..', '_datasets', 'mnist_training_5000x728_equal_classes_binary.h5')
flags['test_data'] = os.path.join('..', '_datasets', 'mnist_test_30x728_equal_classes_binary.h5')

flags['img_shape'] = (28, 28)
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

flags['training_batch_size'] = 5000
flags['max_training_batch_size'] = 1000  # This must be the minibatch size
flags['max_prediction_batch_size'] = flags['training_batch_size']  # This is how big is X at prediction time
flags['approx_num_images_per_side'] = 20
flags['learning_rate'] = 0.001
flags['noise_variance'] = 0.01
flags['fixed_noise_variance'] = True
flags['temperature'] = 0.1
flags['num_iterations'] = 5000  # One iteration corresponds to one full epoch
flags['eval_interval'] = 100
flags['fixed_X_pretraining_iterations'] = 1000
flags['ckpt_interval'] = flags['num_iterations']

# Number of samples to average for each latent point to generate a  greyscale image.
flags['num_samples_to_average'] = 100

###############################################################################
# Generalisation Test
###############################################################################
flags['test_generalisation_batch_size'] = 30
flags['test_generalisation_learning_rate'] = 0.01
flags['test_generalisation_log_var_scaling'] = 0.1
flags['test_generalisation_num_iterations'] = 50
flags['test_generalisation_num_runs'] = 100
flags['test_generalisation_num_samples_to_average'] = 100
flags['test_generalisation_method'] = 'random_training'

###############################################################################
# Plots
###############################################################################
flags['xlim'] = flags['ylim'] = (-3, 3)  # Latent space ranges
flags['delta_ls'] = 0.3  # Latent image sample grid step size
flags['delta_lp'] = 0.01  # Latent plot grid step size
flags['lp_plotter_fig_ext'] = '.png'

# Label colors (palette from D3.js)
flags['label_color_dict'] = {
    0: '#1f77b4',
    1: '#ff7f0e',
    2: '#2ca02c',
    3: '#d62728',
    4: '#9467bd',
    5: '#8c564b',
    6: '#e377c2',
    7: '#7f7f7f',
    8: '#bcbd22',
    9: '#17becf'
}

###############################################################################
# Output Folders
###############################################################################
flags['output_dir'] = 'output'
flags['ckpt_dir'] = os.path.join(flags['output_dir'], 'checkpoints')
flags['ckpt_file'] = os.path.join(flags['ckpt_dir'], 'trained_model')
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
