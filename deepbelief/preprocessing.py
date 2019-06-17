"""This module provides a few data preprocessing functions."""
import numpy as np
import tensorflow as tf
import deepbelief.config as c


def binarize(array, threshold):
    """Convert the values of and array into binary values.

    Any value that is greater than the threshold becomes a one, otherwise it is
    set to zero.

    Args:
        array: Input array.
        threshold: A number.

    Returns:
        An array of type float_type (see config.py).
    """
    binary = np.zeros_like(array, dtype=c.float_type)
    binary[array > threshold] = 1
    return binary


def scale255to1(array):
    """Scale the values of an array by a factor of 1/255.

    Args:
        array: Input array.

    Returns:
        An array of type float_type (see config.py).
    """
    assert np.min(array) >= 0 and np.max(array) <= 255, \
        'The input array values should lie in range [0, 255].'

    array = array.astype(c.float_type)
    array /= 255.0

    return array


def normalize(array):
    """Normalize the values of an array values in range [0, 1].

    Note that the input array is modified in place, and the input elements are
    assumed to be floating point numbers. Also, if the input array is constant
    (i.e., all its elements are the same), the output will be an array of zeros.

    Args:
        array: Input array (modified in place).

    Returns:
        Normalized array.
    """
    arr_min = np.min(array)
    arr_max = np.max(array)

    arr_range = arr_max - arr_min

    if arr_range == 0:
        # Return an array of zeros
        array -= arr_max
        return array

    scaling_factor = 1.0 / arr_range

    array -= arr_min
    array *= scaling_factor
    return array


def standardize(array,
                mean_fname=None,
                std_fname=None,
                return_means_stds=False):
    """Standardize an array.

    Subtract the mean and divide by the standard deviation column-wise. If the
    standard deviation of a column is less or equal to 1e-8 it is set to 1.0.

    Args:
        array: 2-dimensional array.
        mean_fname: Optional file name for the output means.
        std_fname: Optional file name for the output standard deviations.
        return_means_stds: If True, also return the means and stds.

    Returns:
        A standardized array.
    """
    # TODO: Remove saving to file
    assert array.ndim == 2, 'The array must be 2-dimensional'
    assert array.shape[0] > 1, 'The array should contain at least two rows'

    array = array.astype(c.float_type)

    means = array.mean(axis=0)
    stds = array.std(axis=0)

    stds[stds <= 1e-8] = 1.0

    # zero_std_mask = stds == 0.0
    # non_zero_stds = stds[np.logical_not(zero_std_mask)]
    # non_zero_std_mean = non_zero_stds.mean()
    #
    # stds[zero_std_mask] = non_zero_std_mean

    array -= means
    array /= stds

    if mean_fname:
        np.save(mean_fname, means)
    if std_fname:
        np.save(std_fname, stds)

    if return_means_stds:
        return array, means, stds

    return array


def unstandardize(tensor,
                  mean_fname=None,
                  std_fname=None,
                  means=None,
                  stds=None):
    """Un-standardize a tensor.

    Multiply each column of a 2-dimensional tensor by the corresponding
    standard deviation coefficient and then add the mean term.

    Args:
        tensor: 2-dimensional tensor.
        mean_fname: File name for the input means.
        std_fname: File name for the input standard deviations.
        means: Array for the input means.
        stds: Array for the input standard deviations.

    Returns:
        An un-standardized tensor.
    """
    # TODO: Add unit tests
    # TODO: Add shape checks
    # TODO: Remove loading from file
    # TODO: Remove constants, use placeholders instead
    assert type(means) is np.ndarray or mean_fname is not None
    assert type(stds) is np.ndarray or std_fname is not None
    np_means = np.load(mean_fname) if mean_fname else means
    np_stds = np.load(std_fname) if std_fname else stds
    means = tf.constant(np_means, dtype=c.float_type)
    stds = tf.constant(np_stds, dtype=c.float_type)
    mul = tf.multiply(tensor, stds)
    add = tf.add(mul, means)
    return add


def PCA(X, num_components):
    assert X.shape[1] >= num_components, \
        'The number of principal components must be less than or equal ' + \
        'to the dimensionality of the input matrix'
    x_mean = X.mean(0)
    X_zero = X - x_mean
    cov = X_zero.T.dot(X_zero)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    indices = np.argsort(eigenvals)[::-1]
    V = eigenvecs[:, indices]
    V = V[:, :num_components]
    return X_zero.dot(V)


def inverse_softplus(x):
    return tf.log(tf.exp(x) - 1.0)
