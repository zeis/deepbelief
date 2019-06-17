import os
import pickle
import numpy as np
from scipy import misc
import tensorflow as tf
from deepbelief.layers.data import Data
from deepbelief.layers.sbm_lower import SBM_Lower
from deepbelief.layers.gprbm import GPRBM
from deepbelief.layers import gplvm
from deepbelief.util import init_logging
import deepbelief.config as c

# Import local flags.py module without specifying absolute path
import importlib.util
module_spec = importlib.util.spec_from_file_location('flags', 'flags.py')
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
flags = module.flags

# Note, this script should be run after the model is trained (run train.py first).
# Basically, the generalisation experiment consist in trying to "reconstruct" the
# test data as closely as possible. The output of this script will be two Numpy arrays,
# one simply containing the test data and the other one the corresponding data generated
# by the model.

init_logging(flags['test_log_file'])

test_data = Data(flags['test_data'],
                 shuffle_first=False,
                 batch_size=flags['test_generalisation_batch_size'],
                 log_epochs=flags['data_log_epochs'],
                 name='TestData')

test_data_batch = test_data.next_batch()

x_test = tf.get_variable(name='x_test', initializer=tf.random_normal(shape=(1, flags['q']), dtype=c.float_type))

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)

layer1 = SBM_Lower(session=session,
                   side=flags['img_shape'][0],
                   side_overlap=flags['layer_1_side_overlap'],
                   num_h=flags['layer_1_num_h'],
                   temperature=flags['temperature'],
                   name=flags['layer_1_name'])

layer1.restore(flags['layer_1_ckpt'])

kern = gplvm.SEKernel(session=session,
                      alpha=flags['kernel_alpha'],
                      gamma=flags['kernel_gamma'],
                      ARD=flags['kernel_ard'],
                      Q=flags['q'])
kern.restore(flags['kernel_ckpt'])

layer2 = GPRBM(num_v=flags['layer_1_num_h'],
               num_h=flags['layer_2_num_h'],
               Q=flags['q'],
               N=flags['training_batch_size'],
               eval_flag=True,
               kern=kern,
               noise_variance=flags['noise_variance'],
               fixed_noise_variance=flags['fixed_noise_variance'],
               temperature=flags['temperature'],
               bottom=layer1,
               x_test_var=x_test,
               session=session,
               name=flags['layer_2_name'])

layer2.restore(flags['layer_2_ckpt'])

layer2.build_model()

learning_rate = flags['test_generalisation_learning_rate']
optimizer = tf.train.GradientDescentOptimizer(learning_rate=flags['test_generalisation_learning_rate'])

table_path = os.path.join(flags['test_generalisation_plots_dir'], 'table')

layer2.test_generalisation(test_data=test_data_batch,
                           output_table_path=table_path,
                           optimizer=optimizer,
                           log_var_scaling=flags['test_generalisation_log_var_scaling'],
                           num_iterations=flags['test_generalisation_num_iterations'],
                           num_runs=flags['test_generalisation_num_runs'],
                           num_samples_to_average=flags['test_generalisation_num_samples_to_average'])

with open(table_path, 'rb') as f:
    table = pickle.load(f)

assert len(table) == test_data_batch.shape[0]
min_dists = []
num_pixels = flags['img_shape'][0] * flags['img_shape'][1]

test_data_arr = np.zeros(shape=(test_data_batch.shape[0], num_pixels))
model_data_arr = np.zeros(shape=(test_data_batch.shape[0], num_pixels))

for (test_point_index, distance, x, test_point, model_point, predictive_variance) in table:
    min_dists.append(distance / num_pixels)

    misc.imsave(
        os.path.join(flags['test_generalisation_plots_dir'], 'test_point_' + str(test_point_index) + '.png'),
        np.reshape(test_point, newshape=flags['img_shape']))
    misc.imsave(
        os.path.join(flags['test_generalisation_plots_dir'], 'model_point_' + str(test_point_index) + '.png'),
        np.reshape(model_point, newshape=flags['img_shape']))

    test_data_arr[test_point_index] = test_point
    model_data_arr[test_point_index] = model_point

np.save(os.path.join(flags['test_generalisation_plots_dir'], 'test_data'), test_data_arr)
np.save(os.path.join(flags['test_generalisation_plots_dir'], 'model_data'), model_data_arr)

print('Scaled mininum distance average: {}'.format(np.average(min_dists)))
print('Scaled mininum distance std: {}'.format(np.std(min_dists)))

session.close()
