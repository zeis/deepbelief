import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
from deepbelief.layers.data import Data
from deepbelief.layers.rbm import RBM
from deepbelief.layers.gprbm import GPRBM
from deepbelief.layers import gplvm
from deepbelief.plotting import LatentPointPlotter
from deepbelief.plotting import LatentSamplePlotter
from deepbelief.util import init_logging
from deepbelief import preprocessing
import atexit

# Import local flags.py module without specifying absolute path
import importlib.util
module_spec = importlib.util.spec_from_file_location('flags', 'flags.py')
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
flags = module.flags

load_model = False
save_model = True

init_logging(flags['training_log_file'])

lp_plotter = LatentPointPlotter(
    xlim=flags['xlim'],
    ylim=flags['ylim'],
    delta=flags['delta_lp'],
    label_color_dict=flags['label_color_dict'],
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
             W_std=flags['layer_1_W_std'],
             temperature=flags['temperature'],
             name=flags['layer_1_name'])

layer1.init_variables()

layer2 = RBM(session=session,
             num_v=flags['layer_1_num_h'],
             num_h=flags['layer_2_num_h'],
             W_std=flags['layer_2_W_std'],
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
               W_std=flags['layer_3_W_std'],
               max_training_batch_size=flags['max_training_batch_size'],
               max_prediction_batch_size=flags['max_prediction_batch_size'],
               approx_num_images_per_side=flags['approx_num_images_per_side'],
               pretrain_iterations=flags['fixed_X_pretraining_iterations'],
               Q=flags['q'],
               X0=X0,
               N=flags['training_batch_size'],
               V=Y,
               eval_flag=False,
               Y_labels=Y_labels,
               kern=kern,
               noise_variance=flags['noise_variance'],
               fixed_noise_variance=flags['fixed_noise_variance'],
               temperature=flags['temperature'],
               bottom=layer2,
               latent_point_plotter=lp_plotter,
               latent_sample_plotter=ls_plotter,
               session=session,
               name=flags['layer_3_name'])

layer3.init_variables()

layer3.build_model()

optimizer = tf.train.AdamOptimizer(learning_rate=flags['learning_rate'])

summary_writer = tf.summary.FileWriter(flags['summary_writer_file'], session.graph)

saver = tf.train.Saver()

if load_model:
    saver.restore(sess=session, save_path=flags['ckpt_file'])


def save():
    if save_model:
        print("saving model..")
        saver.save(sess=session, save_path=flags['ckpt_file'])


atexit.register(save)


def callback(epoch):
    pass


layer3.optimize(optimizer=optimizer,
                callback=callback,
                num_iterations=flags['num_iterations'],
                eval_interval=flags['eval_interval'],
                ckpt_interval=flags['ckpt_interval'],
                ckpt_dir=flags['ckpt_file'],
                summary_writer=summary_writer)

save()

grid_x = ls_plotter.grid

pred_var = layer3.predict_variance(x=grid_x)

print("pred_var", pred_var.shape)

# Do not close the session, otherwise the atexit saving funciton will fail
# session.close()
