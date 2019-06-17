import unittest
import tensorflow as tf
import numpy as np
from deepbelief.layers import gplvm
import deepbelief.config as c


class TestSEKernel(unittest.TestCase):
    def setUp(self):
        self.session = tf.Session()

        self.kern = gplvm.SEKernel(session=self.session)

        shape = (8, 3)
        self.X = tf.constant(np.random.uniform(size=shape), dtype=c.float_type)
        self.Y = tf.constant(np.random.uniform(size=shape), dtype=c.float_type)

        self.session.run(tf.global_variables_initializer())

    def tearDown(self):
        self.session.close()

    def test_K_single_input(self):
        k = self.kern.K(self.X)
        res = self.kern.K(self.X, self.X)
        k_val, res_val = self.session.run([k, res])
        assert np.allclose(k_val, res_val)

    def test_K_single_input_diag(self):
        k = self.kern.K(self.X, diag=True)
        res = self.kern.K(self.X, self.X, diag=True)
        k_val, res_val = self.session.run([k, res])
        assert np.allclose(k_val, res_val)

    def test_K_diag(self):
        res = tf.diag_part(self.kern.K(self.X, self.Y))
        k = self.kern.K(self.X, self.Y, diag=True)
        k_val, res_val = self.session.run([k, res])
        assert np.allclose(k_val, res_val)

    def test_K_single_input_shape(self):
        k = self.kern.K(self.X)
        res = self.kern.K(self.X, self.X)
        k_val, res_val = self.session.run([k, res])
        assert k_val.shape == res_val.shape

    def test_K_single_input_diag_shape(self):
        k = self.kern.K(self.X, diag=True)
        res = self.kern.K(self.X, self.X, diag=True)
        k_val, res_val = self.session.run([k, res])
        assert k_val.shape == res_val.shape

    def test_K_diag_shape(self):
        res = tf.diag_part(self.kern.K(self.X, self.Y))
        k = self.kern.K(self.X, self.Y, diag=True)
        k_val, res_val = self.session.run([k, res])
        assert k_val.shape == res_val.shape


if __name__ == '__main__':
    unittest.main()
