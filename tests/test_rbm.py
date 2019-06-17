import unittest
import tensorflow as tf
import numpy as np
from deepbelief.layers.rbm import RBM
import deepbelief.config as c


class TestRBM(unittest.TestCase):
    def setUp(self):
        self.NUM_V = 6
        self.NUM_H = 5
        self.BATCH_SIZE = 20
        self.V_BATCH_SHAPE = (self.BATCH_SIZE, 6)
        self.H_BATCH_SHAPE = (self.BATCH_SIZE, 5)
        self.W_STD = 0.01
        self.GRAD_ERROR = 1e-3

        self.sess = tf.Session()

        self.rbm = RBM(num_v=self.NUM_V,
                       num_h=self.NUM_H,
                       W_std=self.W_STD,
                       session=self.sess)

        self.h_batch = tf.constant(
            np.random.binomial(1,
                               0.5,
                               size=self.H_BATCH_SHAPE),
            dtype=c.float_type)

        self.v_batch = tf.constant(
            np.random.binomial(1,
                               0.5,
                               size=self.V_BATCH_SHAPE),
            dtype=c.float_type)

        h_probs = self.rbm.h_probs(self.v_batch)
        self.sample_h = self.rbm.sample_h(h_probs)

        v_probs = self.rbm.v_probs(self.h_batch)
        self.sample_v = self.rbm.sample_v(v_probs)

        self.free_energy = self.rbm.free_energy(self.v_batch)

        self.sess.run(tf.global_variables_initializer())

    def tearDown(self):
        tf.reset_default_graph()
        self.sess.close()

    def test_sample_h_shape(self):
        self.assertEqual(
            self.sess.run(self.sample_h).shape, self.H_BATCH_SHAPE)

    def test_sample_h_values(self):
        values = [0., 1.]
        self.assertTrue(np.all(np.in1d(self.sess.run(self.sample_h), values)))

    def test_sample_v_shape(self):
        self.assertEqual(
            self.sess.run(self.sample_v).shape, self.V_BATCH_SHAPE)

    def test_sample_v_values(self):
        values = [0., 1.]
        self.assertTrue(np.all(np.in1d(self.sess.run(self.sample_v), values)))

    def test_free_energy_shape(self):
        free_energy_arr = self.sess.run(self.free_energy)
        self.assertTrue(free_energy_arr.shape, [self.BATCH_SIZE, 1])

    def test_free_energy_W_grad(self):
        with self.sess as sess:
            error = tf.test.compute_gradient_error(
                self.rbm.W, self.rbm.W.shape.as_list(), self.free_energy,
                self.free_energy.shape.as_list())
            self.assertLess(error, self.GRAD_ERROR)

    def test_free_energy_b_grad(self):
        with self.sess as sess:
            error = tf.test.compute_gradient_error(
                self.rbm.b, self.rbm.b.shape.as_list(), self.free_energy,
                self.free_energy.shape.as_list())
            self.assertLess(error, self.GRAD_ERROR)

    def test_free_energy_c_grad(self):
        with self.sess as sess:
            error = tf.test.compute_gradient_error(
                self.rbm.c, self.rbm.c.shape.as_list(), self.free_energy,
                self.free_energy.shape.as_list())
            self.assertLess(error, self.GRAD_ERROR)


if __name__ == '__main__':
    unittest.main()
