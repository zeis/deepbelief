import os
import pickle
import numpy as np
from scipy import misc
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
from deepbelief.layers.data import Data
from deepbelief.layers.rbm import RBM
from deepbelief.layers.gprbm import GPRBM
from deepbelief.layers import gplvm
from deepbelief.util import init_logging
from deepbelief import preprocessing
import deepbelief.config as c

# Import local flags.py module without specifying absolute path
import importlib.util
module_spec = importlib.util.spec_from_file_location('flags', 'flags.py')
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
flags = module.flags

load_model = False
save_model = True

init_logging(flags['test_log_file'])

test_data = Data(flags['test_data'],
                 has_labels=False,
                 shuffle_first=False,
                 batch_size=flags['test_generalisation_batch_size'],
                 log_epochs=flags['data_log_epochs'],
                 name='TestData')

test_data_batch = test_data.next_batch()

x_test = tf.get_variable(name='x_test', initializer=tf.random_normal(shape=(1, flags['q']), dtype=c.float_type))

training_data = Data(flags['training_data'],
                     has_labels=True,
                     shuffle_first=flags['shuffle_data'],
                     batch_size=flags['training_batch_size'],
                     log_epochs=flags['data_log_epochs'],
                     name='TrainingData')

Y, Y_labels = training_data.next_batch()
print("Y", Y.shape)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)

layer1 = RBM(session=session,
             num_v=flags['img_shape'][0] * flags['img_shape'][1],
             num_h=flags['layer_1_num_h'],
             temperature=flags['temperature'],
             name=flags['layer_1_name'])

layer1.init_variables()

layer2 = RBM(session=session,
             num_v=flags['layer_1_num_h'],
             num_h=flags['layer_2_num_h'],
             temperature=flags['temperature'],
             bottom=layer1,
             name=flags['layer_2_name'])

layer2.init_variables()

kern = gplvm.SEKernel(session=session,
                      alpha=flags['kernel_alpha'],
                      gamma=flags['kernel_gamma'],
                      ARD=flags['kernel_ard'],
                      Q=flags['q'])

# X0 = np.random.multivariate_normal(
#     [0] * flags['q'],
#     np.identity(flags['q']),
#     flags['training_batch_size'])

X0 = preprocessing.PCA(Y, flags['q'])


layer3 = GPRBM(num_v=flags['layer_2_num_h'],
               num_h=flags['layer_3_num_h'],
               max_training_batch_size=flags['max_training_batch_size'],
               max_prediction_batch_size=flags['max_prediction_batch_size'],
               approx_num_images_per_side=flags['approx_num_images_per_side'],
               pretrain_iterations=1,
               Q=flags['q'],
               X0=X0,
               N=flags['training_batch_size'],
               V=Y,
               eval_flag=True,
               Y_labels=Y_labels,
               kern=kern,
               noise_variance=flags['noise_variance'],
               fixed_noise_variance=flags['fixed_noise_variance'],
               temperature=flags['temperature'],
               bottom=layer2,
               x_test_var=x_test,
               session=session,
               name=flags['layer_3_name'])

# layer3.init_variables()

# layer3.build_model()

var_list = tf.global_variables()
var_list.remove(x_test)
saver = tf.train.Saver(var_list)
# saver = tf.train.Saver()
saver.restore(sess=session, save_path=flags['ckpt_file'])

learning_rate = flags['test_generalisation_learning_rate']
optimizer = tf.train.GradientDescentOptimizer(learning_rate=flags['test_generalisation_learning_rate'])

table_path = os.path.join(flags['test_generalisation_plots_dir'], 'table')

layer3.test_generalisation(test_data=test_data_batch,
                           output_table_path=table_path,
                           optimizer=optimizer,
                           log_var_scaling=flags['test_generalisation_log_var_scaling'],
                           num_iterations=flags['test_generalisation_num_iterations'],
                           method=flags['test_generalisation_method'],
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
