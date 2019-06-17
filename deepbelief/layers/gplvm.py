import numpy as np
import tensorflow as tf
from . import layer
from deepbelief import preprocessing
from deepbelief import dist
import deepbelief.config as c


class SEKernel(layer.Stateful):

    def __init__(self,
                 alpha=1.0,
                 gamma=1.0,
                 ARD=False,
                 Q=None,
                 session=None,
                 name='SEKernel'):
        # Squared exponential kernel. Set 'alpha' or 'gamma' to 'False' not
        # to use them as optimization parameters and treat them as
        # constants (1.0).

        self.Q = Q

        if ARD and gamma:
            gamma_shape = (1, self.Q)
            if type(gamma) == np.ndarray:
                assert gamma.shape == gamma_shape
            gamma = gamma * np.ones(shape=gamma_shape, dtype=c.float_type)

        with tf.variable_scope(name):
            if alpha is False:
                self.alpha = tf.constant(1.0, dtype=c.float_type)
            else:
                alpha_init = tf.constant(alpha, dtype=c.float_type)
                alpha_init = preprocessing.inverse_softplus(alpha_init)
                self.opt_alpha = tf.Variable(alpha_init,
                                             dtype=c.float_type,
                                             name='opt_alpha')
                self.alpha = tf.nn.softplus(self.opt_alpha)

            if gamma is False:
                self.gamma = tf.constant(1.0, dtype=c.float_type)
            else:
                gamma_init = tf.constant(gamma, dtype=c.float_type)
                gamma_init = preprocessing.inverse_softplus(gamma_init)
                self.opt_gamma = tf.Variable(gamma_init,
                                             dtype=c.float_type,
                                             name='opt_gamma')
                self.gamma = tf.nn.softplus(self.opt_gamma)

        tf.summary.scalar(name + '_alpha', self.alpha)
        if ARD:
            tf.summary.tensor(name + '_gamma', self.gamma)
        else:
            tf.summary.scalar(name + '_gamma', self.gamma)

        params = {'alpha': self.alpha, 'gamma': self.gamma}

        super().__init__(params=params, session=session, name=name)

    def _sq_dist(self, x, y=None, diag=False):
        """Square distance"""
        keepdims = False if diag else True
        if diag and y is None:
            batch_size = tf.shape(x)[0]
            return tf.zeros(shape=(batch_size, ), dtype=c.float_type)
        xx = x / self.gamma
        sq_xx = tf.square(xx)
        sum_xx = tf.reduce_sum(sq_xx, 1, keepdims=keepdims)
        if y is None:
            yy = xx
            sum_yy = sum_xx
        else:
            yy = y / self.gamma
            sq_yy = tf.square(yy)
            sum_yy = tf.reduce_sum(sq_yy, 1, keepdims=keepdims)
        if diag:
            prod_xy = 2.0 * tf.reduce_sum(xx * yy, 1, keepdims=keepdims)
        else:
            prod_xy = 2.0 * tf.matmul(xx, yy, transpose_b=True)
            sum_yy = tf.transpose(sum_yy)
        return -prod_xy + sum_xx + sum_yy

    def K(self, x, y=None, diag=False):
        sq_dist = self._sq_dist(x, y, diag)
        return self.alpha * tf.exp(-0.5 * sq_dist)


class GPLVM(layer.Layer):

    def __init__(self,
                 Y,
                 Q,
                 Y_labels=None,
                 X0=None,
                 kern=None,
                 noise_variance=1.0,
                 fixed_noise_variance=False,  # If True, the noise variance is constant
                 noise_jitter=1e-6,
                 bottom=None,
                 x_test_var=None,
                 subtract_Y_mean=True,
                 uppropagate_Y=True,  # Ignored if bottom is None
                 latent_point_plotter=None,
                 latent_sample_plotter=None,
                 latent_space_explorer=None,
                 session=None,
                 name='GPLVM'):

        assert type(Y) == np.ndarray, 'Y must be of type numpy.ndarray'
        assert Y.ndim == 2, 'Y must be 2-dimensional numpy.ndarray'

        self.Q = Q
        self.Y_labels = Y_labels
        self.subtract_Y_mean = subtract_Y_mean

        if uppropagate_Y and bottom is not None:
            self.Y = bottom.upprop(tf.constant(Y, dtype=c.float_type))
        else:
            self.Y = tf.constant(Y, dtype=c.float_type)

        if self.subtract_Y_mean:
            self.Y_mean = tf.reduce_mean(self.Y, axis=0, keepdims=True)
            self.Y = self.Y - self.Y_mean

        self.Y_np = session.run(self.Y)

        if X0 is not None:
            self.X0 = X0
        else:
            self.X0 = preprocessing.PCA(self.Y_np, self.Q)
            self.X0 = preprocessing.standardize(self.X0)

        assert self.X0.shape[0] == self.Y_np.shape[0]
        assert self.X0.shape[1] == self.Q

        self.N = self.X0.shape[0]
        self.D = self.Y_np.shape[1]

        with tf.variable_scope(name):
            self.noise_jitter = None
            if fixed_noise_variance is True:
                self.noise_variance = tf.constant(noise_variance, dtype=c.float_type)
            else:
                noise_variance = tf.constant(noise_variance, dtype=c.float_type)
                noise_variance = preprocessing.inverse_softplus(noise_variance)
                self.opt_noise_variance = tf.Variable(noise_variance,
                                                      dtype=c.float_type,
                                                      name='opt_noise_variance')

                self.noise_jitter = noise_jitter
                self.noise_variance = tf.nn.softplus(self.opt_noise_variance)

            self.X = tf.Variable(self.X0, dtype=c.float_type, name='X')

            # TODO: Add a label variable and make use of it at test time. See GPRBM.

        self.kern = kern or SEKernel(session=session, Q=self.Q)

        if x_test_var is not None:
            self.x_test = x_test_var
            self.x_ph = self.x_test
        else:
            self.x_ph = tf.placeholder(shape=(None, self.Q),
                                       dtype=c.float_type,
                                       name='x_ph')

        self.lp_plotter = latent_point_plotter
        self.ls_plotter = latent_sample_plotter
        self.latent_space_explorer = latent_space_explorer

        self.eval_interval = None
        self.loss_values_fname = None
        self.loss_values = None
        self.downprop_op = None

        tf.summary.histogram(name + '_X', self.X)
        tf.summary.scalar(name + '_noise_variance', self.noise_variance)

        params = {'X': self.X, 'noise_variance': self.noise_variance}

        super().__init__(params=params,
                         bottom=bottom,
                         session=session,
                         name=name)

    # def _grad_var_list(self):
    #     var_list = [self.opt_noise_variance,
    #                 self.X,
    #                 self.kern.opt_alpha,
    #                 self.kern.opt_gamma]
    #     var_list = var_list + self.bottom._grad_var_list(lower_layers=True)
    #     return var_list
    #
    # def _grad_var_list(self):
    #     var_list = [self.opt_noise_variance,
    #                 self.X,
    #                 self.kern.opt_alpha,
    #                 self.kern.opt_gamma]
    #     return var_list
    #
    # def _grad_var_list(self):
    #     return self.bottom._grad_var_list(lower_layers=True)

    def param_dict(self):
        d = super().param_dict()
        d[self.kern.name] = self.kern.param_dict()
        return d

    def build_model(self):
        self.K = self.build_K()
        self.L = tf.cholesky(self.K)
        self.log_det = self.build_log_det()
        self.loss = self.build_loss()
        self.pred_mean = self.build_pred_mean()
        self.pred_var = self.build_pred_var()

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

    def build_loss(self, summary=True):
        # TODO: Add hyperprior?
        sys2 = tf.matrix_triangular_solve(self.L, self.Y, lower=True)
        reduce_sum = tf.reduce_sum(tf.square(sys2))
        log_det = self.D * self.log_det
        const = self.D * self.N * np.log(2.0 * np.pi)
        loss = 0.5 * (reduce_sum + log_det + const)
        with tf.control_dependencies([tf.assert_type(loss, c.float_type)]):
            loss = tf.identity(loss)
        if summary:
            tf.summary.scalar(self.name + '_loss', loss)
        return loss

    def build_pred_mean(self):
        KXx = self.kern.K(self.X, self.x_ph)
        sys1 = tf.matrix_triangular_solve(self.L, KXx, lower=True)
        sys2 = tf.matrix_triangular_solve(self.L, self.Y, lower=True)
        pred_mean = tf.matmul(sys1, sys2, transpose_a=True)
        if self.subtract_Y_mean:
            pred_mean = pred_mean + self.Y_mean
        # pred_mean = tf.tanh(pred_mean)
        return pred_mean

    def build_pred_mean_thresholded(self, pred_mean):
        thresholded = tf.tanh(pred_mean)
        return thresholded

    def build_pred_mean_normalized(self, pred_mean):
        normalized = tf.div(
            tf.subtract(pred_mean, tf.reduce_min(pred_mean)),
            tf.subtract(tf.reduce_max(pred_mean), tf.reduce_min(pred_mean)))
        return normalized

    def build_pred_sample(self, pred_mean):
        gaussian_noise = tf.random_normal(shape=tf.shape(pred_mean))
        pred_std = tf.sqrt(self.pred_var)
        mul = tf.multiply(pred_std, gaussian_noise)
        sample = tf.add(pred_mean, mul)
        return sample

    def build_pred_sample_normalized(self, pred_mean):
        sample = self.build_pred_sample(pred_mean)
        sample = self.build_pred_mean_normalized(sample)
        return sample

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

    def predict_mean(self, x):
        if x is not np.array:
            x = np.array(x, ndmin=2)
        pred_mean = self.session.run(self.pred_mean, feed_dict={self.x_ph: x})
        return pred_mean

    def predict_variance(self, x):
        if x is not np.array:
            x = np.array(x, ndmin=2)
        pred_var = self.session.run(self.pred_var, feed_dict={self.x_ph: x})
        return pred_var[:, np.newaxis]

    def datapoint_len(self):
        if self.bottom:
            # Assuming that self.bottom has attribute 'num_v'
            return self.bottom.layers[-1].num_v
        return self.D

    def build_downprop_op(self):
        if self.downprop_op == None:
            self.downprop_op = self.pred_mean
            if self.bottom:
                self.downprop_op = self.bottom.downprop(self.downprop_op)

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
            v_batch = self.session.run(self.downprop_op,
                                       feed_dict={self.x_ph: x_batch})
            return np.mean(v_batch, axis=0)
        elif x.shape[0] > 1:
            v_batch = np.zeros(
                [x.shape[0], datapoint_len], dtype=c.float_type)
            for i in range(num_samples_to_average):
                v_batch = v_batch + self.session.run(
                    self.downprop_op, feed_dict={self.x_ph: x})
            return v_batch / num_samples_to_average

    def save_loss_values(self, loss, iteration):
        idx = iteration // self.eval_interval
        self.loss_values[idx] = (iteration, loss)
        np.save(self.loss_values_fname, self.loss_values[:idx + 1])

    def record_summary(self, i):
        if self.summary is not None:
            summary = self.session.run(self.summary)
            self.summary_writer.add_summary(summary, i)

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
        v_vecs = self.downprop_latent(
            self.ls_plotter.grid, self.ls_plotter.num_samples_to_average)
        fig_name = 'latent_samples'
        if iteration is not None:
            fig_name = fig_name + '-' + str(iteration)
        self.ls_plotter.plot(v_vecs, fig_name)

    def explore_latent_space(self):
        assert self.latent_space_explorer is not None
        self.latent_space_explorer.set_sample_callback(
            self.downprop_latent)
        self.latent_space_explorer.set_variance_callback(
            self.predict_variance)
        self.latent_space_explorer.plot()

    def explore_2D_latent_space(self):
        assert self.latent_space_explorer is not None
        self.latent_space_explorer.set_sample_callback(
            self.downprop_latent)
        tmp_lp_plotter = self.lp_plotter
        self.lp_plotter = self.latent_space_explorer
        self.plot_latent_points(iteration=None)
        self.lp_plotter = tmp_lp_plotter

    def _init_optimizer_slots(self, optimizer):
        slots = [optimizer.get_slot(v, name)
                 for v in tf.global_variables()
                 for name in optimizer.get_slot_names()
                 if optimizer.get_slot(v, name) is not None]

        self.session.run(tf.variables_initializer(slots))

        if hasattr(optimizer, '_get_beta_accumulators'):
            self.session.run(
                tf.variables_initializer(optimizer._get_beta_accumulators()))

    def _eval_step(self, i):
        loss = self.session.run(self.loss)
        self.logger.info('Iteration: %d, Loss: %f', i, loss)
        self.logger.debug(self)

        self.record_summary(i)

        if self.loss_values_fname:
            self.save_loss_values(loss, i)

        if self.lp_plotter:
            self.plot_latent_points(i)

        if self.ls_plotter:
            self.plot_latent_samples(iteration=i)

    def optimize(self,
                 optimizer,
                 num_iterations=100,
                 eval_interval=0,
                 ckpt_interval=0,
                 ckpt_dir=None,
                 summary_writer=None,  # Ignore if eval_interval is 0
                 loss_values_fname=None):  # Ignored if eval_interval is 0
        self.init_variables()
        self.kern.init_variables()

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

            if loss_values_fname:
                self.loss_values_fname = loss_values_fname
                num_points = int(np.floor(
                    (num_iterations - i) / eval_interval) + 1)
                self.loss_values = np.zeros(shape=(num_points, 2))

            self._eval_step(i)

        self.session.graph.finalize()

        try:
            for i in range(i + 1, num_iterations + 1):
                self.session.run(min_step)

                if eval_interval and i % eval_interval == 0:
                    self._eval_step(i)

                if ckpt_dir and ckpt_interval and i % ckpt_interval == 0:
                    self.save_all(i, ckpt_dir)
                    self.kern.save(i, ckpt_dir)
        except KeyboardInterrupt:
            self.logger.info("Training interrupted")
            if eval_interval:
                self._eval_step(i)
            if ckpt_dir and ckpt_interval:
                self.save_all(i, ckpt_dir)
                self.kern.save(i, ckpt_dir)

    def test_generalisation(self,
                            test_data,
                            output_table_path,
                            optimizer,
                            log_var_scaling=0.1,
                            num_iterations=1000,
                            num_runs=1):
        test_point_ph = tf.placeholder(shape=(None, test_data.shape[1]), dtype=c.float_type, name='test_point_ph')

        pred_mean = self.build_pred_mean_normalized(self.pred_mean)
        clipped_mean = tf.clip_by_value(self.pred_mean, 0.0, 1.0)

        # loss = dist.cross_entropy(pred_mean, test_point_ph)
        loss = 1 / test_data.shape[1] * dist.cross_entropy(pred_mean, test_point_ph) + log_var_scaling * tf.log(self.pred_var)

        min_step = optimizer.minimize(loss, var_list=[self.x_test])

        num_training_points = self.session.run(self.X).shape[0]

        assert num_runs <= num_training_points

        if num_runs == num_training_points:
            idx = tf.get_variable(initializer=0, name='idx', dtype=tf.int32)
            zero_idx = tf.assign(idx, 0)
            increment_idx = tf.assign(idx, idx + 1)
            update_x_test = tf.assign(self.x_test, tf.reshape(self.X[idx], shape=(1, self.Q)))
        else:
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

            if num_runs == num_training_points:
                self.session.run(zero_idx)

            for run in range(num_runs):
                self.session.run(update_x_test)

                for i in range(num_iterations):
                    self.session.run(min_step, feed_dict={test_point_ph: test_point})

                distance, x, model_point, clipped_model_point = self.session.run(
                        [loss, self.x_test, pred_mean, clipped_mean],
                        feed_dict={test_point_ph: test_point})

                if num_runs == num_training_points:
                    self.session.run(increment_idx)

                distance = distance[0]
                if distance < prev_dist:
                    if table and table[-1][0] == test_point_index:
                        del table[-1]
                    table.append((test_point_index, distance, x, test_point, model_point, clipped_model_point))
                    prev_dist = distance

                self.logger.info('Test point: {}, Run: {}, Distance: {}'.format(test_point_index, run, distance))

        import pickle
        with open(output_table_path, 'wb') as f:
            pickle.dump(table, f)
