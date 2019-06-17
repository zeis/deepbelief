import tensorflow as tf
import deepbelief.config as c


def mse(predictions, targets):
    """Calculate the mean squared error of two tensors.

    Args:
        predictions: A tensor of datapoints of shape NxD. N is the size of
            the batch.

        targets: A tensor of samples of shape NxD.

    Returns:
        A tensor of shape 1x1.
    """
    sub = tf.subtract(predictions, targets)
    square = tf.square(sub)
    reduce_sum = tf.reduce_sum(square, axis=1)
    mean = tf.reduce_mean(reduce_sum)
    return mean


def cross_entropy(probs, targets, reduce_mean=True):
    small_const = tf.constant(10e-7, dtype=c.float_type)
    probs = tf.clip_by_value(probs, small_const, 1 - small_const)
    logits = tf.log(probs / (1 - probs))
    ce = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=targets)
    if reduce_mean:
        reduce_sum = tf.reduce_sum(ce, axis=1)
        ce = tf.reduce_mean(reduce_sum)
    else:
        ce = tf.reduce_sum(ce)
    return ce
