import numpy as np
import tensorflow as tf
from . import bgrbm
from . import gplvm
from deepbelief import preprocessing
from deepbelief import dist
import deepbelief.config as c


class GPRBM(bgrbm.BGRBM):

    def __init__(self,
                 num_v,  # Number of visible units
                 num_h,  # Number of hidden units
                 Q,  # Dimensionality of X latent space
                 N,  # Number of training points
                 eval_flag,  # Set this to True at test time
                 X0=None,
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
                 loss_plotter=None,
                 distr_plotter=None,
                 filter_plotter=None,
                 latent_point_plotter=None,
                 latent_sample_plotter=None,
                 latent_space_explorer=None,
                 session=None,
                 name='GPRBM'):
        self.Q = Q
        self.N = N
        self.D = num_h
        self.Y_labels = Y_labels
        self.x_test = None
        self.lp_plotter = latent_point_plotter
        self.eval_interval = None
        self.downprop_op = None

        if not eval_flag:
            assert V is not None, 'V data must be provided at training time.'
            assert V.shape[0] == self.N
            self.V_data = tf.constant(V, dtype=c.float_type)

        if X0 is not None:
            assert X0.shape == (self.N, self.Q)
            self.X0 = tf.constant(X0, dtype=c.float_type)
        elif not eval_flag:
            X0 = preprocessing.PCA(V, self.Q)
            self.X0 = tf.constant(preprocessing.standardize(X0), dtype=c.float_type)
        else:
            self.X0 = tf.constant(np.zeros(shape=(self.N, self.Q)), dtype=c.float_type)

        with tf.variable_scope(name):
            self.noise_jitter = None
            if fixed_noise_variance == True:
                self.noise_variance = tf.constant(noise_variance, dtype=c.float_type)
            else:
                noise_variance = tf.constant(noise_variance, dtype=c.float_type)
                noise_variance = preprocessing.inverse_softplus(noise_variance)
                self.opt_noise_variance = tf.Variable(noise_variance, dtype=c.float_type, name='opt_noise_variance')

                self.noise_jitter = noise_jitter
                self.noise_variance = tf.nn.softplus(self.opt_noise_variance)

            self.X = tf.get_variable(name='X', initializer=self.X0)

            alpha_h_init_val = np.ones(shape=(1, self.D))
            alpha_h_init = tf.constant(alpha_h_init_val, dtype=c.float_type)
            alpha_h_init = preprocessing.inverse_softplus(alpha_h_init)
            self.opt_alpha_h = tf.get_variable(initializer=alpha_h_init, name='opt_alpha_h')

            # Constrain alpha_h parameter to be positive
            self.alpha_h = tf.nn.softplus(self.opt_alpha_h)

            self.H1_var = tf.get_variable(name='H1',
                                          initializer=tf.zeros(shape=(self.N, self.D), dtype=c.float_type),
                                          trainable=False)

            self.Y_labels_var = tf.get_variable(name='Y_labels',
                                                initializer=tf.zeros(shape=(self.N, 1)),
                                                trainable=False)

        self.kern = kern or gplvm.SEKernel(session=session, Q=self.Q)

        tf.summary.histogram(name + '_X', self.X)
        tf.summary.scalar(name + '_noise_variance', self.noise_variance)
        tf.summary.histogram(name + '_alpha_h', self.alpha_h)

        self.kern.init_variables()

        self.eval_flag_var = tf.get_variable(name='eval_flag', initializer=eval_flag, trainable=False)
        self.eval_flag_op = tf.assign(self.eval_flag_var, tf.logical_not(self.eval_flag_var))
        session.run(tf.variables_initializer([self.eval_flag_var]))
        self.eval_flag = session.run(self.eval_flag_var)

        self.x_ph = tf.cond(self.eval_flag_var, lambda: self._x_ph(x_test_var), lambda: self.X)

        self.K = self.build_K()
        self.L = tf.cholesky(self.K)
        self.X_prior = self.build_X_prior()
        self.log_det = self.build_log_det()
        self.pred_var = self.build_pred_var()

        pred_std = tf.expand_dims(tf.sqrt(self.pred_var), 1)
        self.sigma_h = self.alpha_h * pred_std

        # This must be called after overriding self.sigma_h
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

        del self.params['sigma_h']
        self.params['alpha_h'] = self.alpha_h
        self.params['X'] = self.X
        self.params['noise_variance'] = self.noise_variance

        if self.eval_flag:
            self.H1 = self.H1_var
        else:
            self.H1 = self.upprop(self.V_data)
            # self.H1 = tf.constant(H0, dtype=c.float_type)
            self.update_H1_var = self.H1_var.assign(self.H1)

        tf.summary.histogram(self.name + '_H1', self.H1)

        self.H1_mean, _ = tf.nn.moments(self.H1, axes=[0], keep_dims=True)
        self.H2 = (self.H1 - self.H1_mean) / self.alpha_h

        self.H2_var = tf.get_variable(name='H2', initializer=tf.zeros(shape=(self.N, self.D)), trainable=False)
        self.H1_mean_var = tf.get_variable(name='H1_mean', initializer=tf.zeros(shape=(1, self.D)), trainable=False)

        self.update_H2_var = self.H2_var.assign(self.H2)
        self.update_H1_mean_var = self.H1_mean_var.assign(self.H1_mean)

    def _x_ph(self, x_test_var):
        if x_test_var is not None:
            self.x_test = x_test_var
            return self.x_test
        else:
            return tf.placeholder(dtype=c.float_type, name='x_ph')

    def build_model(self):
        if self.eval_flag == False:
            sample_mean = self.build_sample_mean(self.H2, self.H1_mean)
            self.loss = self.build_loss(sample_mean)

        if self.latent_space_explorer:
            if self.ls_plotter:
                self.logger.warning(
                    'LatentSpaceExplorer does not work with LatentSamplePlotter, the latter will be disabled.')
                self.ls_plotter = None

            self.pred_mean = self.build_sample_mean(self.H2_var, self.H1_mean_var)

        if self.ls_plotter:
            self.pred_mean = self.build_sample_mean(self.H2_var, self.H1_mean_var)

        if not self.eval_flag and self.Y_labels is not None:
            # Save the labels if provided at training time
            self.session.run(self.Y_labels_var.assign(self.Y_labels))

        if self.eval_flag:
            assert self.session.run(
                tf.is_variable_initialized(self.opt_alpha_h)), 'Restore the variables before calling build_model()'

            self.Y_labels = self.session.run(self.Y_labels_var)
            if not self.Y_labels.any():
                # If self.Y_labels contains all zeros then no labels will be used
                self.Y_labels = None

            self.session.run([self.update_H2_var, self.update_H1_mean_var])

    def build_K(self):
        var = self.noise_variance
        if self.noise_jitter is not None:
            var += self.noise_jitter
        K = self.kern.K(self.X)
        I = tf.diag(tf.ones(self.N, dtype=c.float_type))
        return K + var * I

    def build_log_det(self):
        # See Rasmussen and Williams book A.18
        return 2.0 * tf.reduce_sum(tf.log(tf.diag_part(self.L)))

    def build_X_prior(self):
        return tf.reduce_sum(tf.square(self.X))

    def build_pred_var(self):
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

    def build_sample_mean(self, H, H_mean, num_samples=None):
        KXx = self.kern.K(self.X, self.x_ph)
        sys1 = tf.matrix_triangular_solve(self.L, KXx, lower=True)
        sys2 = tf.matrix_triangular_solve(self.L, H, lower=True)
        H = tf.matmul(sys1, sys2, transpose_a=True)
        H = H * self.alpha_h + H_mean

        gaussian_noise = tf.random_normal(shape=(tf.shape(H)[0], self.num_h))
        if num_samples is not None:
            gaussian_noise = tf.random_normal(shape=(num_samples, self.num_h))
        mul = tf.multiply(self.sigma_h, gaussian_noise)
        H = tf.add(H, mul)
        return H

    def build_gp_loss(self, summary=True):
        sys2 = tf.matrix_triangular_solve(self.L, self.H2, lower=True)
        reduce_sum = tf.reduce_sum(tf.square(sys2))
        log_det = self.D * self.log_det
        X_prior = self.build_X_prior()
        # alpha_prior = tf.square(tf.log(self.kern.alpha))
        # gamma_prior = tf.square(tf.log(self.kern.gamma))
        # sigma_prior = tf.square(tf.log(self.noise_variance))
        loss = 0.5 * (reduce_sum + log_det) + X_prior
        if summary:
            tf.summary.scalar('gp_loss', loss)
        return loss

    def build_loss(self, sample_mean, summary=True):
        V_sample = self.downprop(sample_mean)
        cross_entropy_loss = dist.cross_entropy(V_sample, self.V_data, reduce_mean=False)
        gplvm_loss = self.build_gp_loss(summary=True)
        loss = cross_entropy_loss + gplvm_loss
        with tf.control_dependencies([tf.assert_type(loss, c.float_type)]):
            loss = tf.identity(loss)
        if summary:
            tf.summary.scalar(self.name + '_loss', loss)
        return loss

    def datapoint_len(self):
        return self.layers[-1].num_v

    def build_downprop_op(self):
        if self.downprop_op == None:
            self.downprop_op = self.downprop(self.pred_mean)

    def downprop_latent(self, x, num_samples_to_average=100):
        # TODO: This code is common with other layers
        """Down-propagate the predictive mean a number of times and
        average the result.
        """
        assert type(x) == np.ndarray
        assert x.ndim == 1 or x.ndim == 2
        if x.ndim == 1:
            x = x[np.newaxis, :]

        datapoint_len = self.datapoint_len()
        self.build_downprop_op()

        if x.shape[0] == 1:
            x_batch = np.tile(x, (num_samples_to_average, 1))
            v_batch = self.session.run(self.downprop_op, feed_dict={self.x_ph: x_batch})
            return np.mean(v_batch, axis=0)
        elif x.shape[0] > 1:
            v_batch = np.zeros([x.shape[0], datapoint_len], dtype=c.float_type)
            for i in range(num_samples_to_average):
                v_batch = v_batch + self.session.run(self.downprop_op, feed_dict={self.x_ph: x})
            return v_batch / num_samples_to_average

    def plot_latent_points(self, iteration=None):
        assert self.lp_plotter is not None
        X, var_diag = self.session.run([self.X, self.pred_var], feed_dict={self.x_ph: self.lp_plotter.grid})
        log_var_diag = np.log(var_diag)
        fig_name = 'latent_points'
        if iteration is not None:
            fig_name = fig_name + '-' + str(iteration)
        self.lp_plotter.plot(X, log_var_diag, fig_name, self.Y_labels)

    def plot_latent_samples(self, iteration=None):
        # TODO: This code is common with other layers
        assert self.ls_plotter is not None
        v_vecs = self.downprop_latent(self.ls_plotter.grid, self.ls_plotter.num_samples_to_average)
        fig_name = 'latent_samples'
        if iteration is not None:
            fig_name = fig_name + '-' + str(iteration)
        self.ls_plotter.plot(v_vecs, fig_name)

    def _log_training_info(self, loss, i):
        self.logger.info('Iteration: %d, Loss: %f', i, loss)
        self.logger.debug(self)
        for layer in self.layers:
            if layer != self:
                layer.logger.debug(layer)

    def record_summary(self, i):
        if self.summary is not None:
            summary = self.session.run(self.summary)
            self.summary_writer.add_summary(summary, i)

    def _eval_step(self, i):
        loss = self.session.run(self.loss)

        self._log_training_info(loss, i)

        self.record_summary(i)

        if self.lp_plotter:
            self.plot_latent_points(i)

        if self.ls_plotter:
            self.session.run([self.update_H2_var, self.update_H1_mean_var])

            if self.eval_flag_op is not None:
                self.session.run(self.eval_flag_op)

            self.plot_latent_samples(i)

            if self.eval_flag_op is not None:
                self.session.run(self.eval_flag_op)

    def predict_variance(self, x):
        if x is not np.array:
            x = np.array(x, ndmin=2)
        pred_var = self.session.run(self.pred_var, feed_dict={self.x_ph: x})
        return pred_var[:, np.newaxis]

    def explore_latent_space(self):
        assert self.latent_space_explorer is not None
        self.latent_space_explorer.set_sample_callback(self.downprop_latent)
        self.latent_space_explorer.set_variance_callback(self.predict_variance)
        self.latent_space_explorer.plot()

    def explore_2D_latent_space(self):
        assert self.latent_space_explorer is not None
        self.latent_space_explorer.set_sample_callback(self.downprop_latent)
        tmp_lp_plotter = self.lp_plotter
        self.lp_plotter = self.latent_space_explorer
        self.plot_latent_points(iteration=None)
        self.lp_plotter = tmp_lp_plotter

    def save_all(self, i, ckpt_dir):
        self.session.run(self.update_H1_var)
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
                 num_iterations=100,
                 fixed_X_num_iterations=None,
                 eval_interval=0,
                 ckpt_interval=0,
                 ckpt_dir=None,
                 summary_writer=None):
        self.init_variables()
        self.kern.init_variables()

        var_list = tf.trainable_variables()
        var_list.remove(self.X)
        fixed_X_min_step = optimizer.minimize(
            self.loss,
            var_list=var_list,
            gate_gradients=tf.train.Optimizer.GATE_GRAPH)

        min_step = optimizer.minimize(
            self.loss,
            # var_list=self._grad_var_list(),
            gate_gradients=tf.train.Optimizer.GATE_GRAPH)

        self._init_optimizer_slots(optimizer)

        if eval_interval:
            self.eval_interval = eval_interval

            if summary_writer:
                self.summary_writer = summary_writer
                self.summary = tf.summary.merge_all()

            # Before training, evaluate model at the very start
            i = 0

            self._eval_step(i)

        def iteration_loop(min_step, start_iteration, num_iterations):
            for i in range(start_iteration, num_iterations):
                self.session.run(min_step)

                if eval_interval and i % eval_interval == 0:
                    self._eval_step(i)

                if ckpt_dir and ckpt_interval and i % ckpt_interval == 0:
                    self.save_all(i, ckpt_dir)
                    self.kern.save(i, ckpt_dir)

        self.session.graph.finalize()

        try:
            if fixed_X_num_iterations:
                assert fixed_X_num_iterations <= num_iterations
                iteration_loop(fixed_X_min_step, 1, fixed_X_num_iterations + 1)
                iteration_loop(min_step, fixed_X_num_iterations + 1, num_iterations + 1)
            else:
                iteration_loop(min_step, 1, num_iterations + 1)
        except KeyboardInterrupt:
            self.logger.info("Training interrupted")
            if eval_interval:
                self._eval_step(i)
            if ckpt_dir and ckpt_interval:
                self.save_all(i, ckpt_dir)
                self.kern.save(i, ckpt_dir)

    def test_held_out_data(self,
                           test_data,
                           output_x_fname,
                           optimizer,
                           num_iterations=1000):
        # Note: 'test_data' is assumed to be numpy.array
        test_data_const = tf.constant(test_data, dtype=c.float_type)

        self.session.run(tf.variables_initializer([self.x_test]))

        predict_mean = self.build_sample_mean(self.H2_var, self.H1_mean_var)
        downprop_mean = self.downprop(predict_mean)

        mse = dist.mse(test_data_const, downprop_mean)

        min_step = optimizer.minimize(
            mse,
            var_list=[self.x_test])

        self._init_optimizer_slots(optimizer)

        for i in range(1, num_iterations + 1):
            self.session.run(min_step)

        x = self.session.run(self.x_test)

        np.save(output_x_fname, x)

    def test_generalisation(self,
                            test_data,
                            output_table_path,
                            optimizer,
                            num_iterations=1000,
                            log_var_scaling=0.1,
                            method='all_training',
                            num_runs=1,
                            num_samples_to_average=1):
        """
        :param test_data: A Numpy array of the test data.
        :param output_table_path: Path of the pickled output table.
        :param optimizer: A Tensorflow optimiser.
        :param num_iterations: Number of optimisation iterations per run.
        :param log_var_scaling: Scaling factor of the log predictive variance term.
        :param method: String, it specifies how to initialise the test point at each run.
                       It can be 'all_training', 'random_normal', 'random_training'.
        :param num_runs: Number of random initialisation. If this is equal to X.shape[0] all
                         initial location in X will be tried.
        :param num_samples_to_average: Number of samples to average per point.

        The output of this method is a pickled table. See code.
        """
        test_point_ph = tf.placeholder(shape=(None, test_data.shape[1]), dtype=c.float_type, name='test_point_ph')

        predict_mean = self.build_sample_mean(self.H2_var, self.H1_mean_var, num_samples=num_samples_to_average)
        downprop_mean = self.downprop(predict_mean)

        # downprop_mean = t_interp_predictive_image
        # loss = dist.cross_entropy(tf.reduce_mean(downprop_mean, axis=0, keepdims=True), test_point_ph)
        loss = 1 / test_data.shape[1] * dist.cross_entropy(tf.reduce_mean(downprop_mean, axis=0, keepdims=True),
                                                           test_point_ph) + log_var_scaling * tf.log(self.pred_var)

        min_step = optimizer.minimize(loss, var_list=[self.x_test])

        num_training_points = self.session.run(self.X).shape[0]

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

            for run in range(num_runs):
                self.session.run(update_x_test)

                for i in range(num_iterations):
                    self.session.run(min_step, feed_dict={test_point_ph: test_point})

                distance, x, model_point, predictive_variance = self.session.run(
                    [loss, self.x_test, downprop_mean, self.pred_var],
                    feed_dict={test_point_ph: test_point})

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

    def create_observed_prediction_with_variance(self, tf_input, num_samples):
        # This function returns image predictions and the associate predicted variance for tf_input locations.
        # so tf_input = num_points * Q  therefore t_interp_predicted_image is num_points * (image_side * image_side)
        # and t_interp_hstar_variance is num_points vector

        def build_pred_var_interp(t_interp_X):
            # Return the diagonal of the predictive covariance matrix
            KXx = self.kern.K(self.X, t_interp_X)
            sys1 = tf.matrix_triangular_solve(self.L, KXx, lower=True)
            reduce_sum = tf.reduce_sum(tf.square(sys1), axis=0)
            Kxx = self.kern.K(t_interp_X, diag=True)
            pred_var = Kxx - reduce_sum
            with tf.control_dependencies([tf.assert_non_negative(pred_var)]):
                pred_var = tf.identity(pred_var)
            # pred_var = tf.Print(pred_var, [pred_var])
            return pred_var

        def build_sample_mean_interp(H, H_mean, t_interp_X, num_samples=None):
            KXx = self.kern.K(self.X, t_interp_X)
            sys1 = tf.matrix_triangular_solve(self.L, KXx, lower=True)
            sys2 = tf.matrix_triangular_solve(self.L, H, lower=True)
            H = tf.matmul(sys1, sys2, transpose_a=True)
            H = H * self.alpha_h + H_mean

            gaussian_noise = tf.random_normal(shape=(tf.shape(H)[0], self.num_h))
            if num_samples is not None:
                gaussian_noise = tf.random_normal(shape=(num_samples, self.num_h))
            mul = tf.multiply(self.sigma_h, gaussian_noise)
            H = tf.add(H, mul)
            return H

        t_interp_hstar_variance = build_pred_var_interp(t_interp_X=tf_input)

        num_points = self.session.run(tf.shape(tf_input))[0]
        tf_output_list = []
        for i in range(num_points):
            pred_std = tf.sqrt(t_interp_hstar_variance)[i]
            # Note: overriding self.sigma_h here because build_sample_mean_interp and downprop use it.
            self.sigma_h = self.alpha_h * pred_std
            predict_mean = build_sample_mean_interp(self.H2_var,
                                                    self.H1_mean_var,
                                                    t_interp_X=tf.expand_dims(tf_input[i, :], axis=0),
                                                    num_samples=num_samples)
            downprop_mean = self.downprop(predict_mean)
            tf_output_list.append(tf.reduce_mean(downprop_mean, axis=0, keepdims=True))

        t_interp_predicted_image = tf.concat(tf_output_list, axis=0)

        return t_interp_predicted_image, t_interp_hstar_variance

    def interp_test(self,
                    starting_point_index,
                    end_point_index,
                    num_points=20,
                    interp_lambda=0.1,
                    learning_rate=0.01,
                    num_iterations=500,
                    num_sample_to_average=100):

        X = self.session.run(self.X)
        x_s = np.reshape(X[starting_point_index], newshape=(1, self.Q))
        x_e = np.reshape(X[end_point_index], newshape=(1, self.Q))

        # Returns a column vector of interpolation point scaling numbers (excluding 0.0 and 1.0)
        a = np.linspace(0.0, 1.0, num_points + 2)[1:-1, np.newaxis]

        # Adding 0.0 and 1.0
        a = np.insert(a, 0, 0)
        a = np.append(a, 1.0)
        a = np.reshape(a, newshape=(num_points + 2, 1))

        # Actual interpolation latent points, shape = [num_points, dim_point]
        xx = x_s + np.dot(a, x_e - x_s)

        t_interp_lambda = tf.get_variable(initializer=interp_lambda, name='t_interp_lambda', dtype=c.float_type)
        t_interp_X = tf.get_variable(initializer=tf.zeros(shape=(num_points + 2, self.Q)), name='t_interp_X',
                                     dtype=c.float_type)
        t_interp_X_start = tf.get_variable(initializer=tf.zeros(shape=(1, self.Q)), name='t_interp_X_start',
                                           dtype=c.float_type)
        t_interp_X_end = tf.get_variable(initializer=tf.zeros(shape=(1, self.Q)), name='t_interp_X_end',
                                         dtype=c.float_type)

        self.session.run(tf.assign(t_interp_X_start, x_s))
        self.session.run(tf.assign(t_interp_X_end, x_e))
        self.session.run(tf.assign(t_interp_X, xx))
        self.session.run(tf.assign(t_interp_lambda, interp_lambda))

        t_interp_dist_tmp = tf.concat([t_interp_X[0:1, :] - t_interp_X_start,
                                       t_interp_X[1:, :] - t_interp_X[:-1, :],
                                       t_interp_X_end - t_interp_X[-1:, :]], axis=0)

        t_interp_dist = tf.reduce_mean(tf.reduce_sum(tf.square(t_interp_dist_tmp), axis=1))

        t_interp_predicted_image, t_interp_hstar_variance = self.create_observed_prediction_with_variance(
            tf_input=t_interp_X, num_samples=num_sample_to_average)

        t_interp_objective = t_interp_dist + t_interp_lambda * tf.reduce_mean(tf.log(t_interp_hstar_variance))

        interp_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            t_interp_objective, var_list=[t_interp_X])

        def initialize_uninitialized(sess):
            global_vars = tf.global_variables()
            is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))

        initialize_uninitialized(self.session)

        obs_init = self.session.run(t_interp_predicted_image)

        refresh_iter = int(np.ceil(num_iterations / 10))
        for i in range(num_iterations):
            opt, cost = self.session.run((interp_optimizer, t_interp_objective))
            if (i % refresh_iter) == 0:
                print('  opt iter {:5}: {}'.format(i, cost))

        print('Final iter {:5}: {}'.format(i, cost))

        u_res = self.session.run(t_interp_X)
        u_init = xx
        geodesic_points = self.session.run(t_interp_predicted_image)
        linear_points = obs_init

        return linear_points, geodesic_points
