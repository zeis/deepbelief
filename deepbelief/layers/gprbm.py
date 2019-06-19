import numpy as np
import tensorflow as tf
from . import layer
from deepbelief.util import initialize_uninitialized
from . import gplvm
from deepbelief import preprocessing
from deepbelief import dist
import deepbelief.config as c
from deepbelief.util import batch_index_groups
import keras
from scipy.misc import imresize

class GPRBM(layer.Layer):

    def __init__(self,
                 num_v,  # Number of visible units
                 num_h,  # Number of hidden units
                 Q,  # Dimensionality of X latent space
                 N,  # Number of training points
                 eval_flag,  # Set this to True at test time
                 X0,
                 # Initial value for X. Not needed at test time.
                 # If not provided at training time, it is calculated using PCA on the data.
                 V=None,  # High-dimensional data batch. Not needed at test time.
                 Y_labels=None,  # Data labels. Unused at test time.
                 kern=None,  # Default: SE kernel
                 noise_variance=1.0,  # Initial value of the noise variance
                 fixed_noise_variance=False,  # If True, the noise variance is constant
                 noise_jitter=1e-6,
                 W_std=0.001,  # Initial value for the W weights
                 lr=0.001,  # Learning rate
                 temperature=None,  # Concrete units temperature
                 bottom=None,  # Bottom layer
                 x_test_var=None,
                 latent_point_plotter=None,
                 latent_sample_plotter=None,
                 latent_space_explorer=None,
                 session=None,
                 max_training_batch_size=256,
                 max_prediction_batch_size=1024,
                 approx_num_images_per_side=30,
                 temp_massive_images=False, # TODO temporary quick fix
                 pretrain_iterations=1500,
                 name='GPRBM'):
        self.Q = Q
        self.N = N
        self.D = num_h
        self.Y_labels = Y_labels
        self.x_test = None
        self.lp_plotter = latent_point_plotter
        self.latent_space_explorer = latent_space_explorer
        self.ls_plotter = latent_sample_plotter
        self.eval_interval = None
        self.downprop_op = None
        self.training_batch_size = min(max_training_batch_size, N)
        self.prediction_batch_size = min(max_prediction_batch_size, N)
        self.V = V
        self.X0 = tf.constant(X0, dtype=c.float_type)
        self.X_pred = tf.placeholder(shape=[None, Q], dtype=c.float_type, name='x_ph')
        self.approx_num_images_per_side = approx_num_images_per_side
        self.temp_massive_images = temp_massive_images
        self.P = V.shape[1]
        self.pretrain_iterations = pretrain_iterations

        assert V is not None
        assert V.shape[0] == self.N
        P = V.shape[1]  # TODO What is the dim of V called?
        self.V_data = tf.placeholder(shape=[None, P], dtype=c.float_type)

        # if X0 is not None:
        #     assert X0.shape == (self.N, self.Q)
        #     self.X0 = tf.constant(X0, dtype=c.float_type)
        # elif not eval_flag:
        #     print("before PCA")
        #     X0 = preprocessing.PCA(V, self.Q)
        #     print("after PCA")
        #     self.X0 = tf.constant(preprocessing.standardize(X0), dtype=c.float_type)
        # else:
        #     self.X0 = tf.constant(np.zeros(shape=(self.N, self.Q)), dtype=c.float_type)

        with tf.variable_scope(name):
            self.noise_jitter = None
            if fixed_noise_variance:
                self.noise_variance = tf.constant(noise_variance, dtype=c.float_type)
            else:
                noise_variance = tf.constant(noise_variance, dtype=c.float_type)
                noise_variance = preprocessing.inverse_softplus(noise_variance)
                self.opt_noise_variance = tf.Variable(noise_variance, dtype=c.float_type, name='opt_noise_variance')

                self.noise_jitter = noise_jitter
                self.noise_variance = tf.nn.softplus(self.opt_noise_variance)

            self.X_all = tf.get_variable(name='X', initializer=self.X0, dtype=c.float_type)
            self.X_batch_indices = tf.placeholder(shape=[None], dtype=tf.int32)
            self.X = tf.gather(self.X_all, self.X_batch_indices)

            alpha_h_init_val = np.ones(shape=(1, self.D))
            alpha_h_init = tf.constant(alpha_h_init_val, dtype=c.float_type)
            alpha_h_init = preprocessing.inverse_softplus(alpha_h_init)
            self.opt_alpha_h = tf.get_variable(initializer=alpha_h_init, name='opt_alpha_h')

            # Constrain alpha_h parameter to be positive
            self.alpha_h = tf.nn.softplus(self.opt_alpha_h)

            self.Y_labels_var = tf.get_variable(name='Y_labels', initializer=tf.zeros(shape=(self.N, 1)))

        self.kern = kern or gplvm.SEKernel(session=session, Q=self.Q)

        tf.summary.histogram(name + '_X', self.X)
        tf.summary.scalar(name + '_noise_variance', self.noise_variance)
        tf.summary.histogram(name + '_alpha_h', self.alpha_h)

        self.kern.init_variables()

        # For some reason tf.cond interpret this as 'False'
        # self.eval_flag_var = tf.get_variable(name='eval_flag', initializer=eval_flag)
        # self.eval_flag_op = tf.assign(self.eval_flag_var, tf.logical_not(self.eval_flag_var))
        # session.run(tf.variables_initializer([self.eval_flag_var]))
        # self.eval_flag = session.run(self.eval_flag_var)

        self.eval_flag_var = tf.constant(eval_flag)
        self.eval_flag = eval_flag

        def _X():
            return self.X

        def _x_ph():
            if x_test_var is not None:
                self.x_test = x_test_var
                return self.x_test
            else:
                return tf.placeholder(dtype=c.float_type, name='x_ph')

        self.x_ph = tf.cond(self.eval_flag_var, true_fn=_x_ph, false_fn=_X)
        # self.x_ph = tf.Print(self.x_ph, [tf.shape(self.x_ph)], message='--->')

        self.K = self.build_K()
        self.L = tf.cholesky(self.K)
        self.log_det = self.build_log_det()
        self.pred_var = self.build_pred_var_training()

        pred_std = tf.expand_dims(tf.sqrt(self.pred_var), 1)
        self.sigma_h = self.alpha_h * pred_std

        # This must be called after overriding self.sigma_h
        ## BEGIN base class expansion and reduction ################################################
        if not hasattr(self, 'sigma_h'):
            with tf.variable_scope(name):
                sigma_h_init = tf.ones(shape=(1, num_h), dtype=c.float_type)
                sigma_h_init = preprocessing.inverse_softplus(sigma_h_init)
                self.opt_sigma_h = tf.get_variable(initializer=sigma_h_init,
                                                   name='opt_sigma_h')
                # Constrain sigma_h parameter to be positive
                self.sigma_h = tf.nn.softplus(self.opt_sigma_h)

        ## BEGIN base class expansion and reduction ################################################
        assert isinstance(num_v, int) and num_v > 1
        assert isinstance(num_h, int) and num_h > 1

        self.num_v = num_v
        self.num_h = num_h
        self.temperature = temperature

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

        # END ################################################################################
        self.params['sigma_h'] = self.sigma_h
        # END ################################################################################

        del self.params['sigma_h']
        self.params['alpha_h'] = self.alpha_h
        self.params['X'] = self.X
        self.params['noise_variance'] = self.noise_variance


        self.H1 = self.upprop(self.V_data)

        tf.summary.histogram(self.name + '_H1', self.H1)

        self.H1_mean, _ = tf.nn.moments(self.H1, axes=[0], keep_dims=True)
        self.H2 = (self.H1 - self.H1_mean) / self.alpha_h
        print("H2", self.H2.get_shape())

    def build_model(self):
        sample_mean = self.build_sample_mean(self.H2, self.H1_mean)
        downprop_op = self.downprop(sample_mean)
        self.downprop_op = downprop_op
        self.loss = self.build_loss(downprop_op)

    def build_K(self):
        var = self.noise_variance
        if self.noise_jitter is not None:
            var += self.noise_jitter
        K = self.kern.K(self.X)
        I = tf.diag(tf.ones_like(tf.diag_part(K)))
        return K + var * I

    def build_log_det(self):
        return tf.reduce_sum(tf.log(tf.diag_part(self.L)))

    def build_X_prior(self):
        return tf.reduce_sum(tf.square(self.X))

    def build_pred_var_training(self):
        # Return the diagonal of the predictive covariance matrix
        KXx = self.kern.K(self.X, self.x_ph)
        sys1 = tf.matrix_triangular_solve(self.L, KXx, lower=True)
        reduce_sum = tf.reduce_sum(tf.square(sys1), axis=0)
        Kxx = self.kern.K(self.x_ph, diag=True)
        pred_var = Kxx - reduce_sum
        with tf.control_dependencies([tf.assert_non_negative(pred_var)]):
            pred_var = tf.identity(pred_var)
        # pred_var = tf.Print(pred_var, [pred_var])
        return pred_var

    def build_pred_var_prediction(self):
        # Return the diagonal of the predictive covariance matrix
        KXx = self.kern.K(self.X, self.X_pred)
        sys1 = tf.matrix_triangular_solve(self.L, KXx, lower=True)
        reduce_sum = tf.reduce_sum(tf.square(sys1), axis=0)
        Kxx = self.kern.K(self.X_pred, diag=True)
        pred_var = Kxx - reduce_sum
        with tf.control_dependencies([tf.assert_non_negative(pred_var)]):
            pred_var = tf.identity(pred_var)
        # pred_var = tf.Print(pred_var, [pred_var])
        return pred_var

    def build_sample_mean(self, H, H_mean, num_samples=None):
        KXx = self.kern.K(self.X, self.x_ph)
        sys1 = tf.matrix_triangular_solve(self.L, KXx, lower=True)
        sys2 = tf.matrix_triangular_solve(self.L, H, lower=True)
        sample_mean = tf.matmul(sys1, sys2, transpose_a=True)
        sample_mean = sample_mean * self.alpha_h + H_mean

        gaussian_noise = tf.random_normal(shape=(tf.shape(sample_mean)[0], self.num_h))
        if num_samples is not None:
            gaussian_noise = tf.random_normal(shape=(num_samples, self.num_h))
        mul = tf.multiply(self.sigma_h, gaussian_noise)
        sample_mean = tf.add(sample_mean, mul)
        return sample_mean

    def build_gp_loss(self, summary=True):
        sys2 = tf.matrix_triangular_solve(self.L, self.H2, lower=True)
        reduce_sum = tf.reduce_sum(tf.square(sys2))
        log_det = self.D * self.log_det
        X_prior = self.build_X_prior()
        # alpha_prior = tf.square(tf.log(self.kern.alpha))
        # gamma_prior = tf.square(tf.log(self.kern.gamma))
        # sigma_prior = tf.square(tf.log(self.noise_variance))

        rescale_N_batch = self.N / tf.constant(self.training_batch_size, dtype=c.float_type)

        print("using prior on X")

        loss = 0.5 * rescale_N_batch * (reduce_sum + log_det + X_prior)
        if summary:
            tf.summary.scalar('gp_loss', loss)
        return loss

    def build_loss(self, downprop_op, summary=True):
        V_sample = downprop_op

        rescale_N_batch = self.N / tf.constant(self.training_batch_size, dtype=c.float_type)

        cross_entropy_loss = rescale_N_batch * dist.cross_entropy(
            V_sample,
            self.V_data,
            reduce_mean=False
        )
        gplvm_loss = self.build_gp_loss(summary=True)

        loss = cross_entropy_loss + gplvm_loss
        with tf.control_dependencies([tf.assert_type(loss, c.float_type)]):
            loss = tf.identity(loss)
        if summary:
            tf.summary.scalar(self.name + '_loss', loss)
        return loss

    def datapoint_len(self):
        return self.layers[-1].num_v

    def downprop_latent(self, x, num_samples_to_average=100):
        # TODO: This code is common with other layers
        """Down-propagate the predictive mean a number of times and
        average the result.
        """
        assert type(x) == np.ndarray
        assert x.ndim == 1 or x.ndim == 2
        if x.ndim == 1:
            x = x[np.newaxis, :]

        datapoint_len = self.V.shape[1]

        if x.shape[0] == 1:
            # TODO Has not been checked for validity using batching
            assert False
            x_batch = np.tile(x, (num_samples_to_average, 1))
            v_batch = self.session.run(self.downprop_op, feed_dict={self.x_ph: x_batch})
            return np.mean(v_batch, axis=0)
        elif x.shape[0] > 1:

            v_batch = np.zeros([x.shape[0], datapoint_len], dtype=c.float_type)
            for i in range(num_samples_to_average):

                batch_indices = np.random.choice(self.N, size=self.prediction_batch_size)

                # TODO: Do we need to run this in two steps?

                H2, H1_mean = self.session.run(
                    [self.H2, self.H1_mean],
                    feed_dict={
                        self.X_batch_indices: batch_indices,
                        self.V_data: self.V[batch_indices]
                    }
                )

                v_batch_evaluation = self.session.run(
                    self.downprop_op,
                    feed_dict={
                        self.x_ph: x,
                        self.X_batch_indices: batch_indices,
                        self.H2: H2,
                        self.H1_mean: H1_mean,
                    }
                )

                if i % 10 == 0:
                    print("downprop_latent: sample {} of {}".format(i, num_samples_to_average))

                v_batch = v_batch + v_batch_evaluation
            return v_batch / num_samples_to_average

    def plot_latent_samples(self, iteration=None):

        # # # TODO this return is temp, remove
        # print("NOT PLOTTING, MAKE IT PLOT")
        # return

        # TODO: This code is common with other layers
        assert self.ls_plotter is not None

        x = self.session.run(
            self.X,
            feed_dict={
                self.X_batch_indices: list(range(self.N))
            }
        )

        assert self.approx_num_images_per_side > 1, "OBS has to > 1 (temp fix)"
        xlim = (np.min(x[:, 0]), np.max(x[:, 0]))
        ylim = (np.min(x[:, 1]), np.max(x[:, 1]))

        def dv(lim): return (lim[1] - lim[0]) / (self.approx_num_images_per_side - 1)

        print("Rebuilding grid..")
        self.ls_plotter.rebuild_grid(dx=dv(xlim), dy=dv(ylim), xlim=xlim, ylim=ylim)
        print("Grid built")

        v_vecs = self.downprop_latent(self.ls_plotter.grid, self.ls_plotter.num_samples_to_average)
        fig_name = 'latent_samples'
        if iteration is not None:
            fig_name = fig_name + '-' + str(iteration)

        self.ls_plotter.plot(v_vecs, fig_name)

    def _eval_step(self, i):
        if self.ls_plotter:
            self.plot_latent_samples(i)

    def predict_variance_batched(self, x, num_samples_to_average):
        if x is not np.array:
            x = np.array(x, ndmin=2)

        pred_var = self.session.run(self.pred_var, feed_dict={
            self.x_ph: x,
            self.X_batch_indices: np.random.choice(self.N, size=self.prediction_batch_size)
        })
        for i in range(num_samples_to_average - 1):
            batch_indices = np.random.choice(self.N, size=self.prediction_batch_size)

            pred_var_eval = self.session.run(self.pred_var, feed_dict={
                self.x_ph: x,
                self.X_batch_indices: batch_indices
            })

            pred_var = pred_var + pred_var_eval

        pred_var_estimate = pred_var / num_samples_to_average

        return pred_var_estimate[:, np.newaxis]

    def predict_variance(self, x):
        if x is not np.array:
            x = np.array(x, ndmin=2)
        pred_var = self.session.run(self.pred_var, feed_dict={
            self.x_ph: x,
            self.X_batch_indices: list(range(self.N))
        })
        return pred_var[:, np.newaxis]

    def save_all(self, i, ckpt_dir):
        super().save_all(i, ckpt_dir)

    def _init_optimizer_slots(self, optimizer):
        slots = [optimizer.get_slot(v, name)
                 for v in tf.global_variables()
                 for name in optimizer.get_slot_names()
                 if optimizer.get_slot(v, name) is not None]

        self.session.run(tf.variables_initializer(slots))

        if hasattr(optimizer, '_get_beta_accumulators'):
            self.session.run(
                tf.variables_initializer(optimizer._get_beta_accumulators()))

    def optimize(self,
                 optimizer,
                 callback,
                 num_iterations=100,
                 eval_interval=0,
                 ckpt_interval=0,
                 ckpt_dir=None,
                 summary_writer=None):
        # self.init_variables()
        # self.kern.init_variables()

        min_step = optimizer.minimize(
            self.loss,
            gate_gradients=tf.train.Optimizer.GATE_GRAPH)

        var_list_without_x = tf.trainable_variables()
        var_list_without_x.remove(self.X_all)

        fixed_X_min_step = optimizer.minimize(
            self.loss,
            var_list=var_list_without_x,
            gate_gradients=tf.train.Optimizer.GATE_GRAPH)

        # self._init_optimizer_slots(optimizer)

        if eval_interval:
            self.eval_interval = eval_interval

            if summary_writer:
                self.summary_writer = summary_writer
                self.summary = tf.summary.merge_all()

        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
        keras.backend.get_session().run(tf.initialize_all_variables())
        self.session.run(tf.tables_initializer())
        initialize_uninitialized(sess=self.session)

        self.session.graph.finalize()

        def run(iterations, step, eval=True):
            try:
                for i in range(iterations):
                    batches = batch_index_groups(batch_size=self.training_batch_size, num_samples=self.N)

                    for j, batch_indices in enumerate(batches):
                        _, loss = self.session.run(
                            (step, self.loss),
                            feed_dict={
                                self.X_batch_indices: batch_indices,
                                self.V_data: self.V[batch_indices]
                            }
                        )
                        print("epoch: {}, batch: {}, loss: {}".format(i, j, loss))

                    if eval and eval_interval and i % eval_interval == 0:
                        self._eval_step(i)

                    callback(i)

            except KeyboardInterrupt:
                self.logger.info("Training interrupted")

        pre_iterations = self.pretrain_iterations
        print("pretraining {} iterations..".format(pre_iterations))
        run(iterations=pre_iterations, step=fixed_X_min_step, eval=False)
        print("training..")
        run(iterations=num_iterations, step=min_step)

    # RBM methods #################################################################

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

    def sample_v(self, v_probs, sample=True, summary=True):
        if self.temperature is not None:
            return self._sample_v_concrete(v_probs=v_probs, summary=summary)
        return self._sample_v(v_probs=v_probs, sample=sample, summary=summary)

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

    # BGRBM methods ################################################################

    def h_net_input(self, v_batch):
        """Return the network input afferent to the hidden units.

        Args:
            v_batch: A tensor of shape NxV, where N is the batch size and V is
                the number of visible units.

        Returns:
            A tensor of shape NxH, where H is the number of hidden units.
        """
        matmul = tf.matmul(v_batch, self.W)
        mul = tf.multiply(self.sigma_h, matmul)
        add = tf.add(mul, self.c)
        return add

    def h_probs(self, v_batch):
        """Return the probability of activation of the hidden units.

        Args:
            v_batch: A tensor of shape NxV, where N is the batch size and V is
                the number of visible units.

        Returns:
            A tensor of shape NxH, where H is the number of hidden units.
        """
        return self.h_net_input(v_batch)

    def sample_h(self, h_probs, sample=True, summary=True):
        """Return a sample from the hidden units.

        Args:
            h_probs: A tensor of probabilities of shape NxH, where N is the
                batch size and H is the number of hidden units.

        Returns:
            A tensor of shape NxH.
        """
        if sample:
            batch_size = tf.shape(h_probs)[0]
            gaussian_noise = tf.random_normal(shape=(batch_size, self.num_h))
            mul = tf.multiply(self.sigma_h, gaussian_noise)
            h_batch = tf.add(h_probs, mul)
        else:
            h_batch = h_probs
        tf.summary.histogram(self.name + '_hidden_activations', h_batch)
        return h_batch

    def v_net_input(self, h_batch):
        """Return the network input afferent to the visible units.

        Args:
            h_batch: A tensor of shape NxH, where N is the batch size and H is
                the number of hidden units.

        Returns:
            A tensor of shape NxV, where V is the number of visible units.
        """
        div = tf.div(h_batch, self.sigma_h)
        transpose1 = tf.transpose(div)
        matmul = tf.matmul(self.W, transpose1)
        add = tf.add(matmul, self.b)
        transpose2 = tf.transpose(add)
        return transpose2

    def test_generalisation(self,
                            test_data,
                            output_table_path,
                            optimizer,
                            num_iterations=1000,
                            log_var_scaling=0.1,
                            method='all_training',  # This can be 'all_training', 'random_normal', 'random_training'
                            num_runs=1,
                            num_samples_to_average=1):

        np.random.seed(0)  # Fix seed for reproducibility

        test_point_ph = tf.placeholder(shape=(None, test_data.shape[1]), dtype=c.float_type, name='test_point_ph')

        # predict_mean = self.build_sample_mean(self.H2_var, self.H1_mean_var, num_samples=num_samples_to_average)
        predict_mean = self.build_sample_mean(self.H2, self.H1_mean, num_samples=num_samples_to_average)
        downprop_mean = self.downprop(predict_mean)

        # downprop_mean = t_interp_predictive_image
        # loss = dist.cross_entropy(tf.reduce_mean(downprop_mean, axis=0, keepdims=True), test_point_ph)
        loss = 1 / test_data.shape[1] * dist.cross_entropy(
            tf.reduce_mean(downprop_mean, axis=0, keepdims=True), test_point_ph) + log_var_scaling * tf.log(self.pred_var)

        min_step = optimizer.minimize(loss, var_list=[self.x_test])

        # num_training_points = self.session.run(self.X).shape[0]
        num_training_points = self.prediction_batch_size

        assert num_runs <= num_training_points

        if method == 'all_training':
            assert num_runs == num_training_points
            idx = tf.get_variable(initializer=0, name='idx', dtype=tf.int32)
            zero_idx = tf.assign(idx, 0)
            increment_idx = tf.assign(idx, idx + 1)
            update_x_test = tf.assign(self.x_test, tf.reshape(self.X[idx], shape=(1, self.Q)))
        elif method == 'random_normal':
            update_x_test = tf.assign(self.x_test, tf.random_normal(shape=(1, self.Q), dtype=c.float_type))
        elif method == 'random_training':
            # At each run set the x_test variable at a random training latent location in X
            random_integer = tf.random_uniform(shape=(1,),
                                               minval=0,
                                               maxval=num_training_points - 1,
                                               dtype=tf.int32,
                                               seed=0,
                                               name=None)
            update_x_test = tf.assign(self.x_test, tf.reshape(self.X[random_integer[0]], shape=(1, self.Q)))

        table = []

        self.session.graph.finalize()

        for test_point_index in range(test_data.shape[0]):
            test_point = np.reshape(test_data[test_point_index], newshape=(1, test_data.shape[1]))
            prev_dist = 10e7

            if method == 'all_training':
                self.session.run(zero_idx)

            batch_indices = np.random.choice(self.N, size=self.prediction_batch_size)

            for run in range(num_runs):
                self.session.run(
                    update_x_test,
                    feed_dict={
                        self.X_batch_indices: batch_indices,
                        self.V_data: self.V[batch_indices]
                    }
                )

                H2, H1_mean = self.session.run(
                    [self.H2, self.H1_mean],
                    feed_dict={
                        self.X_batch_indices: batch_indices,
                        self.V_data: self.V[batch_indices]
                    }
                )

                for i in range(num_iterations):
                    self.session.run(
                        min_step,
                        feed_dict={
                            test_point_ph: test_point,
                            self.X_batch_indices: batch_indices,
                            self.V_data: self.V[batch_indices],
                            self.H2: H2,
                            self.H1_mean: H1_mean
                        }
                    )

                distance, x, model_point, predictive_variance = self.session.run(
                    [loss, self.x_test, downprop_mean, self.pred_var],
                    feed_dict={
                        test_point_ph: test_point,
                        self.X_batch_indices: batch_indices,
                        self.H2: H2,
                        self.H1_mean: H1_mean
                    }
                )

                model_point = np.mean(model_point, axis=0)

                if method == 'all_training':
                    self.session.run(increment_idx)

                distance = distance[0]
                if distance < prev_dist:
                    if table and table[-1][0] == test_point_index:
                        del table[-1]
                    table.append((test_point_index, distance, x, test_point, model_point, predictive_variance[0]))
                    prev_dist = distance

                self.logger.info('Test point: {}, Run: {}, Distance: {}'.format(test_point_index, run, distance))

        import pickle
        with open(output_table_path, 'wb') as f:
            pickle.dump(table, f)