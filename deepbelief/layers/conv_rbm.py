import tensorflow as tf
from . import rbm
import deepbelief.config as c
import numpy as np


class ConvRBM(rbm.RBM):

    def __init__(self,
                 v_side,  # Note: h_side is ceil(v_side / stride)
                 stride=2,
                 filter_side=5,
                 in_filter_channels=1,
                 out_filter_channels=1,
                 W_std=0.001,
                 lr=0.001,
                 temperature=None,
                 bottom=None,
                 loss_plotter=None,
                 distr_plotter=None,
                 filter_plotter=None,
                 latent_sample_plotter=None,
                 latent_space_explorer=None,
                 session=None,
                 name='ConvRBM'):
        assert isinstance(stride, int) and stride > 0

        # TODO: Overwrite non-concrete sampling functions as well
        # TODO: Double check everything

        self.v_side = v_side
        self.strides = [1, stride, stride, 1]

        self.filter_side = filter_side

        self.h_side = int(np.ceil(self.v_side / stride))

        self.in_filter_channels = in_filter_channels
        self.out_filter_channels = out_filter_channels

        with tf.variable_scope(name):
            init_W = tf.truncated_normal(shape=[self.filter_side,
                                                self.filter_side,
                                                self.in_filter_channels,
                                                self.out_filter_channels],
                                         stddev=W_std,
                                         dtype=c.float_type)
            self.W = tf.get_variable(initializer=init_W, name='W')

        num_v = self.v_side * self.v_side
        num_h = self.h_side * self.h_side
        super().__init__(num_v=num_v,
                         num_h=num_h,
                         W_std=W_std,
                         lr=lr,
                         temperature=temperature,
                         bottom=bottom,
                         loss_plotter=loss_plotter,
                         distr_plotter=distr_plotter,
                         filter_plotter=filter_plotter,
                         latent_sample_plotter=latent_sample_plotter,
                         latent_space_explorer=latent_space_explorer,
                         session=session,
                         name=name)

    def h_net_input(self, v_batch):
        batch_size = tf.shape(v_batch)[0]
        reshaped = tf.reshape(v_batch, [batch_size, self.v_side, self.v_side, -1])

        convolved = tf.nn.conv2d(
            input=reshaped,
            filter=self.W,
            strides=self.strides,
            padding='SAME',
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=[1, 1, 1, 1]
        )

        out_shape = tf.shape(convolved)

        if self.out_filter_channels == 1:
            reshaped2 = tf.reshape(convolved, [batch_size, out_shape[1] * out_shape[2]])
            h_net_input = tf.add(reshaped2, self.c)
        else:
            reshaped2 = tf.reshape(convolved, [batch_size, out_shape[1] * out_shape[2], -1])
            h_net_input = tf.add(reshaped2, tf.expand_dims(self.c, 2))

        return h_net_input

    def v_net_input(self, h_batch):
        batch_size = tf.shape(h_batch)[0]
        reshaped = tf.reshape(h_batch, [batch_size, self.h_side, self.h_side, -1])

        deconvolved = tf.nn.conv2d_transpose(
            value=reshaped,
            filter=self.W,
            output_shape=[batch_size, self.v_side, self.v_side, self.in_filter_channels],
            strides=self.strides,
            padding='SAME',
            data_format='NHWC'
        )

        if self.in_filter_channels == 1:
            reshaped2 = tf.reshape(deconvolved, [batch_size, self.v_side * self.v_side])
            v_net_input = tf.add(reshaped2, tf.transpose(self.b))
        else:
            reshaped2 = tf.reshape(deconvolved, [batch_size, self.v_side * self.v_side, -1])
            v_net_input = tf.add(reshaped2, tf.expand_dims(tf.transpose(self.b), 2))

        return v_net_input

    def _sample_h_concrete(self, h_probs, summary=True):
        batch_size = tf.shape(h_probs)[0]
        if self.out_filter_channels == 1:
            shape = (batch_size, self.num_h)
        else:
            shape = (batch_size, self.num_h, self.out_filter_channels)
        unif_rand_vals = tf.random_uniform(shape=shape,
                                           minval=0.0,
                                           maxval=1.0,
                                           dtype=c.float_type)
        small_const = tf.constant(10e-7, dtype=c.float_type)
        pre_sigmoid = (tf.log(h_probs + small_const)
                       - tf.log(1.0 - h_probs + small_const)
                       + tf.log(unif_rand_vals + small_const)
                       - tf.log(1.0 - unif_rand_vals + small_const))
        h_batch = tf.sigmoid(1.0 / self.temperature * pre_sigmoid)

        if summary:
            tf.summary.histogram(self.name + '_hidden_activations', h_batch)
        return h_batch

    def _sample_v_concrete(self, v_probs, summary=True):
        batch_size = tf.shape(v_probs)[0]
        if self.in_filter_channels == 1:
            shape = (batch_size, self.num_v)
        else:
            shape = (batch_size, self.num_v, self.in_filter_channels)
        unif_rand_vals = tf.random_uniform(shape=shape,
                                           minval=0.0,
                                           maxval=1.0,
                                           dtype=c.float_type)
        small_const = tf.constant(10e-7, dtype=c.float_type)
        pre_sigmoid = (tf.log(v_probs + small_const)
                       - tf.log(1.0 - v_probs + small_const)
                       + tf.log(unif_rand_vals + small_const)
                       - tf.log(1.0 - unif_rand_vals + small_const))
        v_batch = tf.sigmoid(1.0 / self.temperature * pre_sigmoid)

        if summary:
            tf.summary.histogram(self.name + '_visible_activations', v_batch)
        return v_batch
