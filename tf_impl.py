from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


@tf.custom_gradient
def lambertw(z):
	"""
	Calculate the k=0 branch of lambert w function.

	Inverse of z*exp(z), i.e. w(z)*exp(w(z)) = z

	Iterative algorithm from
	https://www.quora.com/How-is-the-Lambert-W-Function-computed

	Gradient from implicit differentiation.
	"""
	step_tol = 1e-12

	def cond(w, step):
		return tf.greater(tf.reduce_max(tf.abs(step)), step_tol)

	def body(w, step):
		ew = tf.exp(w)
		numer = w*ew - z
		step = numer/(ew*(w+1) - (w+2)*numer/(2*w + 2))
		w = w - step
		return w, step

	w = tf.log(1 + z)
	step = w

	w, step = tf.while_loop(
		cond, body, (w, step), back_prop=False,
		maximum_iterations=20
		)

	def grad_fn(dy):
		print('Calling grad_fn')
		return w / (z*(1 + w))

	return w, grad_fn
