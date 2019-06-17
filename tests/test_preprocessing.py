import unittest
import numpy as np
import deepbelief.preprocessing as preprocessing


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.arr1 = np.random.normal(loc=2, scale=5, size=(5, 5))
        self.arr1[:, 0] = 0.3  # Set every element of column 0 to a constant
        self.arr2 = np.random.randint(256, size=(5, 5))

    def test_binarize_shape(self):
        res = preprocessing.binarize(self.arr1, 0.6)
        self.assertEqual(res.shape, self.arr1.shape)

    def test_binarize_values(self):
        res = preprocessing.binarize(self.arr1, 0.5)
        values = [0., 1.]
        self.assertTrue(np.all(np.in1d(res, values)))

    def test_scale255to1_shape(self):
        res = preprocessing.scale255to1(self.arr2)
        self.assertEqual(res.shape, self.arr1.shape)

    def test_scale255to1_values(self):
        res = preprocessing.scale255to1(self.arr2)
        self.assertTrue((res >= 0).all() and (res <= 1).all())

    def test_standardize_shape(self):
        res = preprocessing.standardize(self.arr1)
        self.assertEqual(res.shape, self.arr1.shape)

    def test_standardize_means(self):
        res = preprocessing.standardize(self.arr1)
        res_means = np.mean(res, axis=0)
        zeros = np.zeros(shape=(1, self.arr1.shape[1]))
        self.assertTrue(np.allclose(res_means, zeros, atol=1e-5))

    def test_standardize_std_zeros(self):
        res = preprocessing.standardize(self.arr1)
        res_stds = np.std(res, axis=0)
        self.assertEqual(res_stds[0], 0.)

    def test_standardize_std_ones(self):
        res = preprocessing.standardize(self.arr1)
        res_stds = np.std(res, axis=0)
        ones = np.ones(shape=(1, self.arr1.shape[1] - 1))
        self.assertTrue(np.allclose(res_stds[1:], ones))


if __name__ == '__main__':
    unittest.main()
