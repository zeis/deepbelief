import unittest
import tensorflow as tf
import numpy as np
from deepbelief.layers.gbrbm import GBRBM
import deepbelief.config as c


class TestGBRBM(unittest.TestCase):
    def setUp(self):
        self.NUM_V = 6
        self.NUM_H = 5
        self.V_BATCH_SHAPE = (20, 6)
        self.H_BATCH_SHAPE = (20, 5)
        self.W_STD = 0.01

        self.session = tf.Session()

        self.rbm = GBRBM(num_v=self.NUM_V,
                         num_h=self.NUM_H,
                         W_std=self.W_STD,
                         session=self.session)
        self.h_batch_ph = tf.placeholder(dtype=c.float_type,
                                         shape=self.H_BATCH_SHAPE)
        self.h_batch = np.random.binomial(1, 0.5, size=self.H_BATCH_SHAPE)
        self.v_batch_ph = tf.placeholder(dtype=c.float_type,
                                         shape=self.V_BATCH_SHAPE)
        self.v_batch = np.random.normal(size=self.V_BATCH_SHAPE)

        self.zeros = np.zeros(shape=self.V_BATCH_SHAPE)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        v_probs = self.rbm.v_probs(self.h_batch_ph)

        self.result = self.sess.run(
            self.rbm.sample_v(v_probs),
            feed_dict={
                self.h_batch_ph: self.h_batch
            })

    def tearDown(self):
        tf.reset_default_graph()
        self.session.close()

    def test_sample_v_shape(self):
        self.assertEqual(self.result.shape, self.V_BATCH_SHAPE)

    def test_sample_v_means(self):
        res_means = np.mean(self.result, axis=0)
        self.assertTrue(np.allclose(res_means, self.zeros, atol=0.1))

    def test_sample_v_stds(self):
        res_stds = np.std(self.result, axis=0)
        self.assertTrue(np.allclose(res_stds, self.zeros, atol=0.1))


if __name__ == '__main__':
    unittest.main()
