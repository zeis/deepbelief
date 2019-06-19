import sys
import logging
import numpy as np
import tensorflow as tf


def init_logging(filename=None,
                 file_level=logging.DEBUG,
                 console_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s] - %(levelname)s - %(name)s - %(message)s')

    if filename:
        fh = logging.FileHandler(filename, mode='w')
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(ch)


def batch_index_groups(batch_size, num_samples):
    # 'a' here corresponds to np.arange(a).
    # 'size' is the length of the output tuple
    # 'replace' set to 'False' ensures that the sample is without replacement. That is, it can be sampled only once.
    indices = np.random.choice(a=num_samples, size=num_samples, replace=False)

    def offset(i): return (i * batch_size) % num_samples

    num_full_batches = int(num_samples / batch_size)

    return [
        indices[offset(i):(offset(i) + batch_size)]
        for i in range(num_full_batches)
    ]


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))