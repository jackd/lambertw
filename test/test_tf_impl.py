#!/usr/bin/python
import unittest
import tensorflow as tf
import scipy.special as ss
from lambertw.tf_impl import lambertw
import numpy as np

# tf.enable_eager_execution()


class LambertWTests(unittest.TestCase):
	def test_value(self):
		z = np.random.uniform(high=10, size=(100,))
		ssw = ss.lambertw(z)
		w = lambertw(z)
		with tf.Session() as sess:
			w = sess.run(w)

		np.testing.assert_allclose(w, ssw)

	def test_gradient(self):
		eps = 1e-4
		# z = tf.random_uniform(shape=(100,), dtype=tf.float32)*10
		z = tf.constant(
			np.random.uniform(high=10, size=(100,)), dtype=tf.float32)
		w = lambertw(z)
		grad, = tf.gradients(w, z)
		grad_fd = (lambertw(z+eps) - w) / eps
		with tf.Session() as sess:
			grad_fd, grad = sess.run((grad, grad_fd))
		np.testing.assert_allclose(grad, grad_fd, atol=1e-2)


if __name__ == '__main__':
	unittest.main()
