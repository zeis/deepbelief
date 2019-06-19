import tensorflow as tf
import numpy as np
from . import layer
from deepbelief import preprocessing
from deepbelief import dist
import deepbelief.config as c


class RBM(layer.Layer):
    """Binary RBM layer.

    The visible and hidden units are binary.

    Args:
        num_v: Number of visible units.
        num_h: Number of hidden units.
        W_std: Weight initialization standard deviation. Default: 0.1.
        lr: Initial learning rate. Default: 0.1.
        temperature: Parameter of Concrete units.
            If None, standard bernoulli units are used. Default: None.
        bottom: A layer object to use as the bottom layer. Default: None.
        loss_plotter: GraphPlotter object. Default: None.
        distr_plotter: ImageRowPlotter object. Default: None.
        filter_plotter: ImageGridPlotter object. Default: None.
        session: Session object.
        name: Name of the layer. Default: 'RBM'.
    """

    def __init__(self,
                 num_v,
                 num_h,
                 W_std=0.1,
                 lr=0.1,
                 temperature=None,
                 bottom=None,
                 loss_plotter=None,
                 distr_plotter=None,
                 filter_plotter=None,
                 latent_sample_plotter=None,
                 latent_space_explorer=None,
                 session=None,
                 name='RBM'):

        assert isinstance(num_v, int) and num_v > 1
        assert isinstance(num_h, int) and num_h > 1

        self.num_v = num_v
        self.num_h = num_h
        self.temperature = temperature
        self.loss_plotter = loss_plotter
        self.distr_plotter = distr_plotter
        self.filter_plotter = filter_plotter
        self.ls_plotter = latent_sample_plotter
        self.latent_space_explorer = latent_space_explorer

        with tf.variable_scope(name):
            if not hasattr(self, 'W'):
                # Weight matrix
                init_W = tf.truncated_normal(shape=(num_v, num_h),
                                             stddev=W_std,
                                             dtype=c.float_type)
                self.W = tf.get_variable(initializer=init_W, name='W')

            # Visible bias vector
            init_b = tf.zeros(shape=(num_v, 1), dtype=c.float_type)
            self.b = tf.get_variable(initializer=init_b, name='b')

            # Hidden bias vector
            init_c = tf.zeros(shape=(1, num_h), dtype=c.float_type)
            self.c = tf.get_variable(initializer=init_c, name='c')

            # Learning rate
            init_lr = tf.constant(lr, dtype=c.float_type)
            self.lr = tf.get_variable(initializer=init_lr,
                                      trainable=False,
                                      name='lr')

            # Checkpoint iteration
            init_checkpoint_iter = tf.constant(1, dtype=c.int_type)
            self.checkpoint_iter = tf.get_variable(
                initializer=init_checkpoint_iter,
                trainable=False,
                name='ckpt_iter')

        # Current iteration
        self.iter = 0

        # Placeholder variables
        self.training_batch_ph = None
        self.test_batch_ph = None
        self.low_dim_batch_ph = None
        self.input_ph = None

        # Tensor variables
        self.downprop_op = None

        # Variables to hold the values of the loss plot
        self.loss_x_vals = None
        self.loss_y_vals = None

        params = {'W': self.W, 'b': self.b, 'c': self.c}

        super().__init__(params, bottom, session, name)

    def h_net_input(self, v_batch):
        """Return the network input afferent to the hidden units.

        Args:
            v_batch: A tensor of shape NxV, where N is the batch size and V is
                the number of visible units.

        Returns:
            A tensor of shape NxH, where H is the number of hidden units.
        """
        matmul = tf.matmul(v_batch, self.W)
        add = tf.add(matmul, self.c)
        return add

    def v_net_input(self, h_batch):
        """Return the network input afferent to the visible units.

        Args:
            h_batch: A tensor of shape NxH, where N is the batch size and H is
                the number of hidden units.

        Returns:
            A tensor of shape NxV, where V is the number of visible units.
        """
        transpose1 = tf.transpose(h_batch)
        matmul = tf.matmul(self.W, transpose1)
        add = tf.add(matmul, self.b)
        transpose2 = tf.transpose(add)
        return transpose2

    def h_probs(self, v_batch):
        """Return the probability of turning on for the hidden units.

        Args:
            v_batch: A tensor of shape NxV, where N is the batch size and V is
                the number of visible units.

        Returns:
            A tensor of shape NxH, where H is the number of hidden units.
        """
        h_net_input = self.h_net_input(v_batch)
        sigmoid = tf.sigmoid(h_net_input)
        return sigmoid

    def v_probs(self, h_batch):
        """Return the probability of turning on for the visible units.

        Args:
            h_batch: A tensor of shape NxH, where N is the batch and H is the
                number of hidden units.

        Returns:
            A tensor of shape NxV, where V is the number of visible units.
        """
        v_net_input = self.v_net_input(h_batch)
        sigmoid = tf.sigmoid(v_net_input)
        return sigmoid

    def sample_h(self, h_probs, sample=True, summary=True):
        if self.temperature is not None:
            return self._sample_h_concrete(h_probs=h_probs, summary=summary)
        return self._sample_h(h_probs=h_probs, sample=sample, summary=summary)

    def sample_v(self, v_probs, sample=True, summary=True):
        if self.temperature is not None:
            return self._sample_v_concrete(v_probs=v_probs, summary=summary)
        return self._sample_v(v_probs=v_probs, sample=sample, summary=summary)

    def _sample_h(self, h_probs, sample=True, summary=True):
        """Return a sample from the hidden units.

        Args:
            h_probs: A tensor of probabilities of shape NxH, where N is the
                size of the batch and H is the number of hidden units.

            sample: A boolean. If False, return the probabilities of turning on
                for the hidden units rather than a sample. Default: True.

        Returns:
            A tensor of shape NxH.
        """
        if sample:
            batch_size = tf.shape(h_probs)[0]
            unif_rand_vals = tf.random_uniform(shape=(batch_size, self.num_h),
                                               minval=0.0,
                                               maxval=1.0,
                                               dtype=c.float_type)
            h_batch = tf.cast(
                tf.greater(h_probs, unif_rand_vals),
                dtype=c.float_type)
        else:
            h_batch = h_probs
        if summary:
            tf.summary.histogram(self.name + '_hidden_activations', h_batch)
        return h_batch

    def _sample_v(self, v_probs, sample=True, summary=True):
        """Return a sample from the visible units.

        Args:
            v_probs: A tensor of probabilities of shape NxV, where N is the
                size of the batch and V is the number of hidden units.

            sample: A boolean. If False, return the probabilities of turning on
                for the visible units rather than a sample. Default: True.

        Returns:
            A tensor of shape NxV.
        """
        if sample:
            batch_size = tf.shape(v_probs)[0]
            unif_rand_vals = tf.random_uniform(shape=(batch_size, self.num_v),
                                               minval=0.0,
                                               maxval=1.0,
                                               dtype=c.float_type)
            v_batch = tf.cast(
                tf.greater(v_probs, unif_rand_vals),
                dtype=c.float_type)
        else:
            v_batch = v_probs
        if summary:
            tf.summary.histogram(self.name + '_visible_activations', v_batch)
        return v_batch

    def _sample_h_concrete(self, h_probs, summary=True):
        batch_size = tf.shape(h_probs)[0]
        unif_rand_vals = tf.random_uniform(shape=(batch_size, self.num_h),
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

        unif_rand_vals = tf.random_uniform(shape=(batch_size, self.num_v),
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

    def upprop(self, v_batch, sample=True, summary=True):
        """Return a sample from the hidden units.

        If there are lower layers connected, the input batch is first fed to
        the lowest layer of the network and up-propagated through the higher
        layers, until the current layer is reached and a sample from the hidden
        units is drawn.

        Args:
            v_batch: A tensor of shape NxV, where N is the batch size and V is
                the number of visible units of the lowest layer.

            sample: A boolean. If False, propagate the probabilities for the
                units to turn on rather than a sample. Default: True.

        Returns:
            A tensor of shape NxH, where H is the number of hidden units of
            the current layer.
        """
        if self.bottom:
            v_batch = self.bottom.upprop(v_batch, sample, summary)
        h_probs = self.h_probs(v_batch)
        h_batch = self.sample_h(h_probs, sample, summary)
        return h_batch

    def downprop(self, h_batch, sample=True, summary=True):
        """Return a sample from the visible units of the lowest layer.

        A sample from the visible units is first drawn from the current layer,
        then, if there is any lower layer connected, the sample is
        down-propagated through the lower layers until the lowest layer
        is reached, and a sample from its visible units is drawn.

        Args:
            h_batch: A tensor of shape NxH, where N is the batch size and H is
                the number of hidden units of the current layer.

            sample: A boolean. If False, propagate the probabilities for the
                units to turn on rather than a sample. Default: True.

        Returns:
            A tensor of shape of NxV, where V is the number of visible units of
            the lowest layer.
        """
        v_probs = self.v_probs(h_batch)
        v_batch = self.sample_v(v_probs, sample, summary)
        if self.bottom:
            v_batch = self.bottom.downprop(v_batch, sample, summary)
        return v_batch

    def propagate(self, v_batch, k=1, sample=True, summary=True):
        """Perform k cycles of 'upprop()' and 'downprop()'.

        Args:
            v_batch: A tensor of shape NxV, where N is the batch size and V is
                the number of visible units of the lowest layer.

            sample: A boolean. If False, propagate the probabilities for the
                units to turn on rather than a sample. Default: True.

        Returns:
            A tensor of shape of NxV.
        """
        assert k > 0, 'k must be > 0'
        k -= 1

        h_batch = self.upprop(v_batch, sample, summary)
        v_batch2 = self.downprop(h_batch, sample, summary)

        if k > 0:
            v_batch2 = self.propagate(v_batch2, k, sample, summary)

        return v_batch2

    def gibbs_sample(self, h_batch, k=1, summary=True):
        """Perform k steps of Gibbs sampling.

        Args:
            h_batch: A tensor of shape NxH, where N is the batch size and H is
                the number of hidden units.

        Returns:
            A tuple containing: the activation probabilities of the visible
            units (shape: NxV), a sample from the visible units (shape: NxV),
            the activation probabilities of the hidden units (shape: NxH), and
            a sample from the hidden units (NxV).
        """
        assert k > 0, 'k must be > 0'
        k -= 1

        v_probs = self.v_probs(h_batch)
        v_batch = self.sample_v(v_probs, summary)
        h_probs = self.h_probs(v_batch)
        h_batch = self.sample_h(h_probs, summary)

        if k > 0:
            v_probs, v_batch, h_probs, h_batch = (
                self.gibbs_sample(h_batch, k))

        return v_probs, v_batch, h_probs, h_batch

    def free_energy(self, v_batch):
        """Calculate the free energy of each datapoint in a batch.

        Args:
            v_batch: A tensor of datapoints of shape NxV.  N is the size of
                the batch and V is the number of visible units.

        Returns:
            A tensor of shape Nx1.
        """
        h_net_input = self.h_net_input(v_batch)
        softplus = tf.nn.softplus(h_net_input)
        matmul = tf.matmul(v_batch, self.b)
        reduce_sum = tf.reduce_sum(softplus,
                                   axis=1,
                                   keepdims=True)
        add2 = tf.add(matmul, reduce_sum)
        neg = tf.negative(add2)
        return neg

    def cd_loss(self, v_0_batch, v_k_batch):
        """Compute the Contrastive Divergence approximation to the negative
        log-likelihood of a batch of datapoints and model reconstructions.

        Args:
            v_0_batch: A tensor of datapoints of shape NxV. N is the size of
                the batch and V is the number of visible units.

            v_k_batch: A tensor of model reconstructions of shape NxV.
        """
        stop_gradient_v = tf.stop_gradient(v_0_batch)
        stop_gradient_k = tf.stop_gradient(v_k_batch)
        free_energy_v = self.free_energy(stop_gradient_v)
        free_energy_k = self.free_energy(stop_gradient_k)
        mean_v = tf.reduce_mean(free_energy_v)
        mean_k = tf.reduce_mean(free_energy_k)
        sub = tf.subtract(mean_v, mean_k)
        return sub

    def build_generate_v_given_h(self, sample=True):
        self.input_ph = tf.placeholder(
            dtype=c.float_type,
            shape=[None, self.num_h])

        v_batch = self.sample_v(self.v_probs(self.input_ph), sample=sample)
        if self.bottom is not None:
            v_batch = self.bottom.downprop(v_batch, sample)

        self.downprop_op = v_batch

    def datapoint_len(self):
        if self.bottom:
            return self.bottom.layers[-1].num_v
        return self.num_v

    def build_downprop_op(self):
        if self.downprop_op is None:
            self.build_generate_v_given_h(sample=True)

    def downprop_latent(self, x, num_samples_to_average=100):
        # TODO: This code is common with other layers
        assert type(x) == np.ndarray
        assert x.ndim == 1 or x.ndim == 2
        if x.ndim == 1:
            x = x[np.newaxis, :]

        datapoint_len = self.datapoint_len()
        self.build_downprop_op()

        if x.shape[0] == 1:
            x_batch = np.tile(x, (num_samples_to_average, 1))
            v_batch = self.session.run(self.downprop_op,
                                       feed_dict={self.input_ph: x_batch})
            return np.mean(v_batch, axis=0)
        elif x.shape[0] > 1:
            v_batch = np.zeros(
                [x.shape[0], datapoint_len], dtype=c.float_type)
            for i in range(num_samples_to_average):
                v_batch = v_batch + self.session.run(
                    self.downprop_op, feed_dict={self.input_ph: x})
            return v_batch / num_samples_to_average

    def plot_latent_samples(self, iteration=None):
        # TODO: This code is common with other layers
        assert self.ls_plotter is not None
        v_vecs = self.downprop_latent(
            self.ls_plotter.grid, self.ls_plotter.num_samples_to_average)
        fig_name = 'latent_samples'
        if iteration is not None:
            fig_name = fig_name + '-' + str(iteration)
        self.ls_plotter.plot(v_vecs, fig_name)

    def test_generalisation(self,
                            test_data,
                            output_x_fname,
                            num_iterations=1000,
                            num_runs=1,
                            num_samples_to_average=100):
        datapoint_ph = tf.placeholder(shape=(1, test_data.shape[1]),
                                      dtype=c.float_type,
                                      name='datapoint_ph')

        tiled_test_datapoint = tf.tile(datapoint_ph, [num_samples_to_average, 1])

        datapoint = tf.get_variable(
            shape=(num_samples_to_average, test_data.shape[1]), name='datapoint')

        init_datapoint = datapoint.assign(tiled_test_datapoint)

        upprop = self.upprop(datapoint, sample=True)
        downprop = self.downprop(upprop, sample=True)
        update_datapoint = datapoint.assign(downprop)

        loss = dist.cross_entropy(tf.reduce_mean(datapoint, axis=0, keepdims=True), datapoint_ph)

        table = []

        for dp in range(test_data.shape[0]):
            print("Datapoint: " + str(dp))
            datapoint_val = np.reshape(test_data[dp], newshape=(1, test_data.shape[1]))
            self.session.run(
                init_datapoint,
                feed_dict={datapoint_ph: datapoint_val})

            prev_dist = 10e7

            for run in range(num_runs):
                for i in range(num_iterations):
                    self.session.run(update_datapoint)

                reconstruction = self.session.run(tf.reduce_mean(downprop, axis=0))

                distance = self.session.run(
                        loss,
                        feed_dict={datapoint_ph: datapoint_val})
                print(distance)

                if distance < prev_dist:
                    if table and table[-1][0] == dp:
                        del table[-1]
                    table.append((dp, distance, datapoint_val, reconstruction))
                    prev_dist = distance

        import pickle
        with open(output_x_fname, 'wb') as f:
            pickle.dump(table, f)

    def explore_latent_space(self):
        self.latent_space_explorer.set_sample_callback(
            self.downprop_latent)
        self.latent_space_explorer.plot()

    def _grad_var_list(self, lower_layers=False):
        var_list = [self.W, self.b, self.c]
        if lower_layers and self.bottom:
            var_list = var_list + self.bottom._grad_var_list(lower_layers=True)
        return var_list

    def _min_step(self, sgd, v_0_batch, v_k_batch):
        loss = self.cd_loss(v_0_batch, v_k_batch)
        return sgd.minimize(loss,
                            var_list=self._grad_var_list(),
                            gate_gradients=tf.train.Optimizer.GATE_GRAPH)

    def _eval_step(self,
                   test_data,
                   test_loss,
                   data_distr=None,
                   model_distr=None):
        test_batch = test_data.next_batch()

        fetches = [test_loss]

        if (self.distr_plotter and data_distr is not None and
                model_distr is not None):
            fetches = fetches + [data_distr]
            fetches = fetches + [model_distr]

        if self.filter_plotter:
            fetches = fetches + [self.W]

        feed_dict = {self.test_batch_ph: test_batch}
        res = self.session.run(fetches, feed_dict=feed_dict)

        # res = self.session.run(fetches,
        #                        feed_dict={self.test_batch_ph: test_batch})

        values = {fetches[i]: res[i] for i in range(len(res))}

        self.logger.info(
            'Iteration: %d, Test loss: %f', self.iter, values[test_loss])
        self.logger.debug(self)

        self.loss_x_vals.append(self.iter)
        self.loss_y_vals.append(values[test_loss])

        if self.loss_plotter:
            loss_fig_name = 'loss_' + self.name
            self.loss_plotter.set_data(self.loss_x_vals, self.loss_y_vals,
                                       loss_fig_name)

        if self.distr_plotter:
            distr_fig_name = 'distr_' + self.name + '_' + str(self.iter)
            self.distr_plotter.set_data(
                (values[data_distr], values[model_distr]), distr_fig_name)

        if self.filter_plotter:
            filter_fig_name = 'filters_' + self.name + '_' + str(self.iter)
            self.filter_plotter.set_data(values[self.W], filter_fig_name)

    def _loss_val_lists(self, start_iter, max_iter, eval_interval):
        # Preallocate empty lists (for efficiency reasons).
        num_points = int(np.floor((max_iter - start_iter) / eval_interval))
        self.loss_x_vals = [] * num_points
        self.loss_y_vals = [] * num_points

    def _test_distrs(self, test_v_batch):
        data_distr = tf.reduce_mean(self.test_batch_ph,
                                    axis=0,
                                    keepdims=True)
        model_distr = tf.reduce_mean(test_v_batch,
                                     axis=0,
                                     keepdims=True)
        return data_distr, model_distr

    def train(self,
              training_data,
              test_data,
              num_gibbs_steps=1,
              pcd=True,
              lr_interval=None,  # Number of iterations before a drop
              lr_drop=None,  # Learning rate drop factor
              test_mean_fname=None,
              test_std_fname=None,
              max_iterations=1000,
              eval_interval=0,
              ckpt_interval=0,
              ckpt_dir=None):
        """Train RBM layer using CD-k or PCD-k."""
        # TODO: Add summary argument and code for parameter summaries
        self.training_batch_ph = tf.placeholder(
            dtype=c.float_type,
            shape=training_data.batch_shape(),
            name='training_batch_ph')
        self.test_batch_ph = tf.placeholder(dtype=c.float_type,
                                            shape=test_data.batch_shape(),
                                            name='test_batch_ph')

        self.init_variables()

        if self.bottom:
            v_0_batch = self.bottom.upprop(self.training_batch_ph)
            test_v_0_batch = self.bottom.upprop(self.test_batch_ph)
        else:
            v_0_batch = self.training_batch_ph
            test_v_0_batch = self.test_batch_ph

        h_0_probs = self.h_probs(v_0_batch)
        h_0_batch = self.sample_h(h_0_probs)

        if pcd:
            # Note, the first batch of datapoints is only used to initialize
            # the h_batch variable, so, it is only up-propagated once and
            # discarded.
            training_batch_arr = training_data.next_batch()
            h_batch = tf.Variable(h_0_batch, trainable=False)
            self.session.run(
                tf.variables_initializer([h_batch]),
                feed_dict={
                    self.training_batch_ph: training_batch_arr
                })
        else:
            h_batch = h_0_batch

        _, v_k_batch, _, h_k_batch = self.gibbs_sample(h_batch,
                                                       k=num_gibbs_steps)

        test_h_0_probs = self.h_probs(test_v_0_batch)
        test_h_0_batch = self.sample_h(test_h_0_probs)
        test_v_1_batch = self.downprop(test_h_0_batch)

        data_distr, model_distr = self._test_distrs(test_v_1_batch)

        if test_mean_fname and test_std_fname:
            data_distr = preprocessing.unstandardize(data_distr,
                                                     test_mean_fname,
                                                     test_std_fname)
            model_distr = preprocessing.unstandardize(model_distr,
                                                      test_mean_fname,
                                                      test_std_fname)

        test_mse = dist.mse(test_v_1_batch, self.test_batch_ph)

        sgd = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        min_step = self._min_step(sgd, v_0_batch, v_k_batch)

        if lr_drop:
            drop_lr = tf.assign(self.lr, tf.multiply(self.lr, lr_drop))

        if ckpt_interval and ckpt_dir:
            update_ckpt_iter = self.checkpoint_iter.assign_add(ckpt_interval)

        start_iter = self.session.run(self.checkpoint_iter)
        max_iter = max_iterations + 1

        if eval_interval:
            self._loss_val_lists(start_iter, max_iter, eval_interval)

            # Evaluate the model the first time before training
            self.iter = start_iter - 1
            self._eval_step(test_data, test_mse, data_distr, model_distr)

        training_ops = [min_step]

        if pcd:
            update_h_batch = h_batch.assign(h_k_batch)
            training_ops.append(update_h_batch)

        for self.iter in range(start_iter, max_iter):
            training_batch_arr = training_data.next_batch()

            self.session.run(training_ops,
                             feed_dict={
                                 self.training_batch_ph: training_batch_arr
                             })

            # Drop learning rate
            if lr_interval and lr_drop and self.iter % lr_interval == 0:
                self.session.run(drop_lr)

            if eval_interval and self.iter % eval_interval == 0:
                self._eval_step(test_data, test_mse, data_distr, model_distr)

            if ckpt_dir and ckpt_interval and self.iter % ckpt_interval == 0:
                self.session.run(update_ckpt_iter)
                self.save(self.iter, ckpt_dir)

    def backprop(self,
                 training_data,
                 test_data,
                 training_sampling=True,
                 training_num_propagation_cycles=5,  # Ignored if training_sampling is 'False'
                 test_num_propagation_cycles=5,
                 learning_rate=0.1,
                 max_iterations=1000,
                 ignore_ckpt_iter=False,
                 eval_interval=0,
                 ckpt_interval=0,
                 ckpt_dir=None):
        self.training_batch_ph = tf.placeholder(
            dtype=c.float_type,
            shape=training_data.batch_shape(),
            name='training_batch_ph')
        self.test_batch_ph = tf.placeholder(dtype=c.float_type,
                                            shape=test_data.batch_shape(),
                                            name='test_batch_ph')

        self.init_variables()

        if not training_sampling:
            training_num_propagation_cycles = 1

        training_v_batch_1 = self.propagate(
            self.training_batch_ph,
            k=training_num_propagation_cycles,
            sample=training_sampling)
        test_v_batch_1 = self.propagate(
            self.test_batch_ph,
            k=test_num_propagation_cycles,
            sample=True)

        training_loss = dist.cross_entropy(
            training_v_batch_1, self.training_batch_ph)
        test_loss = dist.cross_entropy(
            test_v_batch_1, self.test_batch_ph)

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)

        training_step = optimizer.minimize(
            training_loss,
            # var_list=self._grad_var_list(lower_layers=True),
            gate_gradients=tf.train.Optimizer.GATE_GRAPH)

        data_distr, model_distr = self._test_distrs(test_v_batch_1)

        if ckpt_interval and ckpt_dir:
            update_ckpt_iter = self.checkpoint_iter.assign_add(ckpt_interval)

        if ignore_ckpt_iter:
            start_iter = 1
        else:
            start_iter = self.session.run(self.checkpoint_iter)

        max_iter = max_iterations + 1

        if eval_interval:
            self._loss_val_lists(start_iter, max_iter, eval_interval)

            # Evaluate the model the first time before training
            self._eval_step(test_data, test_loss, data_distr, model_distr)

        for self.iter in range(start_iter, max_iter):
            training_batch = training_data.next_batch()
            feed_dict = {self.training_batch_ph: training_batch}

            self.session.run(training_step, feed_dict=feed_dict)

            if eval_interval and self.iter % eval_interval == 0:
                self._eval_step(test_data, test_loss, data_distr, model_distr)

            if ckpt_dir and ckpt_interval and self.iter % ckpt_interval == 0:
                self.session.run(update_ckpt_iter)
                self.save_all(self.iter, ckpt_dir)
