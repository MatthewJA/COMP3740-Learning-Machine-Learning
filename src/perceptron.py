#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classify data using a perceptron.
"""

from __future__ import division

import theano
import numpy

import mnist

class Perceptron(object):
	"""
	Learns to classify and then classifies data.
	"""

	def __init__(self, input_batch, output_batch, input_dimension, output_dimension, learning_rate=0.01):
		"""
		The input dimension is a number representing the dimensions of the input
		vectors.

		The output dimension is the number of labels that the classifier will
		classify.
		"""

		self.initialise_symbolic_input()
		self.initialise_bias_vector()
		self.initialise_weight_matrix()

	def initialise_weight_matrix(self):
		"""
		Initialise the weight matrix and store it in a shared Theano value as self.W.
		"""

		# Initialise a weight matrix W.
		matrix = theano.numpy.zeros(
			(self.input_dimension, self.output_dimension),
			dtype=theano.config.floatX
		)
		# matrix = theano.numpy.random.rand(self.input_dimension, self.output_dimension).astype(theano.config.floatX)

		# Store the weight matrix.
		self.W = theano.shared(
			value=matrix,
			name="W",
			borrow=True
		)

	def initialise_symbolic_input(self):
		"""
		Initialise the symbolic version of the input.
		"""

		self.x = theano.tensor.matrix("x")
		self.y = theano.tensor.ivector("y")

	def initialise_bias_vector(self):
		"""
		Initialise the bias vector and store it in a shared Theano value as self.b.
		"""

		# Initialise a bias vector b.
		vector = theano.numpy.zeros(
			(self.output_dimension,),
			dtype=theano.config.floatX
		)

		# Store the bias vector.
		self.b = theano.shared(
			value=vector,
			name="b",
			borrow=True
		)

		def get_probability_matrix(self):
		"""
		Get symbolic probability matrix.
		"""

		return theano.tensor.nnet.softmax(theano.dot(self.x, self.W) + self.b)

	def initialise_theano_functions(self):
		"""
		Set up Theano symbolic functions and store them in self.
		"""

		gradient_wrt_W = theano.tensor.grad(cost=self.get_negative_log_likelihood(), wrt=self.W)
		gradient_wrt_b = theano.tensor.grad(cost=self.get_negative_log_likelihood(), wrt=self.b)
		updates = [
			(self.W, self.W - self.learning_rate * gradient_wrt_W),
			(self.b, self.b - self.learning_rate * gradient_wrt_b)
		]
		index = theano.tensor.lscalar()
		batch_size = theano.tensor.lscalar()

		self.train_model_once = theano.function(
			inputs=[index, batch_size],
			outputs=self.get_negative_log_likelihood(),
			updates=updates,
			givens={
				self.x: self.input_batch[index*batch_size:(index+1)*batch_size],
				self.y: self.output_batch[index*batch_size:(index+1)*batch_size]
			}
		)

	def train_model_once(self):

