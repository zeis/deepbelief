import os
import json
import logging
import numpy as np
import tensorflow as tf

class Stateful:
    def __init__(self, params=None, session=None, name=None):
        self.params = params
        self.session = session or tf.get_default_session()
        self.name = name

        self.variables = [var
                          for var in tf.global_variables()
                          if var.name.startswith(self.name)]

        self.min_array_size_for_stats = 10

        self.logger = logging.getLogger(self.name)

        if self.variables:
            self.saver = tf.train.Saver(var_list=self.variables, max_to_keep=0)

        self.summary_writer = None
        self.summary = None

    def __str__(self):
        return json.dumps(self.param_dict(), indent=4, sort_keys=True)

    def param_dict(self):
        p_keys = list(self.params.keys())
        p_vals = self.session.run(list(self.params.values()))
        d = {}

        for i in range(len(p_vals)):
            try:
                d[p_keys[i]] = np.asscalar(p_vals[i])
            except ValueError:
                pass

            if isinstance(p_vals[i], np.ndarray):
                if p_vals[i].size > self.min_array_size_for_stats:
                    d[p_keys[i] + '_avg'] = np.asscalar(np.mean(p_vals[i]))
                    d[p_keys[i] + '_min'] = np.asscalar(np.min(p_vals[i]))
                    d[p_keys[i] + '_max'] = np.asscalar(np.max(p_vals[i]))
                    d[p_keys[i] + '_std'] = np.asscalar(np.std(p_vals[i]))
                    d[p_keys[i] + '_min_abs'] = \
                        np.asscalar(np.min(np.abs(p_vals[i])))
                else:
                    d[p_keys[i]] = p_vals[i].tolist()
        return d

    def init_variables(self):
        check_vars_initialized = tf.assert_variables_initialized(
            self.variables)
        try:
            self.session.run(check_vars_initialized)
        except tf.errors.FailedPreconditionError:
            self.session.run(tf.variables_initializer(self.variables))

    def save(self, i, ckpt_dir):
        assert os.path.isdir(ckpt_dir), \
            'The directory does not exist: %s' % ckpt_dir

        ckpt_file = os.path.join(ckpt_dir, self.name)
        if self.variables:
            self.saver.save(self.session, ckpt_file, global_step=i)
        self.logger.info('State saved')

    def restore(self, ckpt_fname):
        if self.variables:
            self.saver.restore(self.session, ckpt_fname)


class Layer(Stateful):
    def __init__(self,
                 params=None,
                 bottom=None,
                 session=None,
                 name='Layer'):

        super().__init__(params=params, session=session, name=name)

        self.bottom = bottom

        layer = self
        self.layers = [layer]
        while layer.bottom:
            self.layers = self.layers + [layer.bottom]
            layer = layer.bottom

    def layer_zero(self):
        layer = self
        while layer.bottom:
            layer = layer.bottom
        return layer

    def save_all(self, i, ckpt_dir):
        for layer in self.layers:
            layer.save(i, ckpt_dir)
