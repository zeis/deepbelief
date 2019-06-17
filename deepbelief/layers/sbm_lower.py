import tensorflow as tf
from . import rbm
import deepbelief.config as c


class SBM_Lower(rbm.RBM):

    def __init__(self,
                 side,
                 side_overlap,
                 num_h,
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
                 name='SBM_Lower'):

        assert num_h % 4 == 0
        assert side % 2 == 0
        assert side_overlap % 2 == 0

        self.side_overlap = side_overlap
        one_fourth_num_h = int(num_h / 4)
        self.side = side

        self.patch_side = int(self.side / 2 + self.side_overlap / 2)
        self._patch_side_squared = self.patch_side * self.patch_side

        with tf.variable_scope(name):
            # Weight matrix shared among the four patches of visible units
            init_W = tf.truncated_normal(shape=(self._patch_side_squared, one_fourth_num_h),
                                         stddev=W_std,
                                         dtype=c.float_type)
            self.W = tf.get_variable(initializer=init_W, name='W')

        num_v = self.side * self.side
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

        self._b_transpose = tf.transpose(self.b)

    def h_net_input(self, v_batch):
        batch_size = tf.shape(v_batch)[0]
        reshaped = tf.reshape(v_batch, [batch_size, self.side, self.side])

        padding_side = self.patch_side - self.side_overlap
        slice1 = tf.slice(reshaped, [0, 0, 0], [batch_size, self.patch_side, self.patch_side])
        slice2 = tf.slice(reshaped, [0, 0, padding_side], [batch_size, self.patch_side, -1])
        slice3 = tf.slice(reshaped, [0, padding_side, padding_side], [batch_size, -1, -1])
        slice4 = tf.slice(reshaped, [0, padding_side, 0], [batch_size, -1, self.patch_side])

        flattened1 = tf.reshape(slice1, [batch_size, self._patch_side_squared])
        flattened2 = tf.reshape(slice2, [batch_size, self._patch_side_squared])
        flattened3 = tf.reshape(slice3, [batch_size, self._patch_side_squared])
        flattened4 = tf.reshape(slice4, [batch_size, self._patch_side_squared])

        matmul1 = tf.matmul(flattened1, self.W)
        matmul2 = tf.matmul(flattened2, self.W)
        matmul3 = tf.matmul(flattened3, self.W)
        matmul4 = tf.matmul(flattened4, self.W)

        h_net_input = tf.concat([matmul1, matmul2, matmul3, matmul4], axis=1)

        return h_net_input

    def v_net_input(self, h_batch):
        split1, split2, split3, split4 = tf.split(h_batch, 4, axis=1)
        # Shape: [batch_size self.num_h / 4]

        W_transpose = tf.transpose(self.W)
        matmul1 = tf.matmul(split1, W_transpose)
        matmul2 = tf.matmul(split2, W_transpose)
        matmul3 = tf.matmul(split3, W_transpose)
        matmul4 = tf.matmul(split4, W_transpose)
        # Shape: [batch_size self.patch_side^2]

        batch_size = tf.shape(h_batch)[0]
        reshaped1 = tf.reshape(matmul1, [batch_size, self.patch_side, self.patch_side])
        reshaped2 = tf.reshape(matmul2, [batch_size, self.patch_side, self.patch_side])
        reshaped3 = tf.reshape(matmul3, [batch_size, self.patch_side, self.patch_side])
        reshaped4 = tf.reshape(matmul4, [batch_size, self.patch_side, self.patch_side])
        # Shape: [batch_size self.patch_side self.patch_side]

        padding_side = self.patch_side - self.side_overlap
        paddings1 = tf.constant([[0, 0], [0, padding_side], [0, padding_side]])
        padded1 = tf.pad(reshaped1, paddings1, 'CONSTANT')
        paddings2 = tf.constant([[0, 0], [0, padding_side], [padding_side, 0]])
        padded2 = tf.pad(reshaped2, paddings2, 'CONSTANT')
        paddings3 = tf.constant([[0, 0], [padding_side, 0], [padding_side, 0]])
        padded3 = tf.pad(reshaped3, paddings3, 'CONSTANT')
        paddings4 = tf.constant([[0, 0], [padding_side, 0], [0, padding_side]])
        padded4 = tf.pad(reshaped4, paddings4, 'CONSTANT')

        overlapped = padded1 + padded2 + padded3 + padded4

        flattened = tf.reshape(overlapped, [batch_size, self.num_v])

        v_net_input = tf.add(flattened, self._b_transpose)

        return v_net_input
