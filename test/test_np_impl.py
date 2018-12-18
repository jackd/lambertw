#!/usr/bin/python
import unittest
import scipy.special as ss
from lambertw.np_impl import lambertw, lambertw_grad
import numpy as np


class LambertWTests(unittest.TestCase):
	def test_value(self):
		z = np.random.uniform(high=10, size=(100,))
		ssw = ss.lambertw(z)
		w = lambertw(z)
		np.testing.assert_allclose(w, ssw, atol=1e-5)

	def test_gradient(self):
		step = 1e-8
		eps = 1e-8
		z = np.random.uniform(high=10, size=(100,))
		w = lambertw(z, step)
		grad = lambertw_grad(z, w)
		grad_fd = (lambertw(z+eps) - w) / eps
		np.testing.assert_allclose(grad, grad_fd, atol=1e-5)


if __name__ == '__main__':
	unittest.main()
