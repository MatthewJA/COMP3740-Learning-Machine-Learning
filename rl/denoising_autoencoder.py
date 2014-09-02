#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Learns features of inputs.

Principal Author: Matthew Alger
"""

from __future__ import division

import time

import theano
import numpy

class Denoising_Autoencoder(object):
	"""
	It like learns how to take noisy versions of the input to unnoisy versions
	of the input like back to the original versions of the input. -- Buck
	"""

	def __init__(self, input_dimension, hidden_dimension, symbolic_input=None, rng=None, learning_rate=0.01):
		"""
		input_dimension: The dimension of the input vectors.
		hidden_dimension: How many hidden nodes to map to.
		symbolic_input: Optional. A symbolic input value.
		rng: Optional. A NumPy RandomState.
		"""

		if symbolic_input is None:
			self.initialise_symbolic_input()

		if rng is None:
			self.initialise_rng()

		self.activation = theano.tensor.tanh
		self.learning_rate = learning_rate
		self.initialise_parameters()
		self.initialise_theano_functions()

	def initialise_parameters(self):
		"""
		Initialises and subsequently stores a weight matrix, bias vector,
		and reverse bias vector.
		"""

		low = -numpy.sqrt(6/(self.input_dimension + self.hidden_dimension))
		high = numpy.sqrt(6/(self.input_dimension + self.hidden_dimension))
		if self.activation is theano.tensor.nnet.sigmoid:
			# We know the optimum distribution for tanh and sigmoid, so we
			# assume that we're using tanh unless we're using sigmoid.
			low *= 4
			high *= 4

		self.weights = theano.shared(
			value=numpy.asarray(
				self.rng.uniform( # This distribution is apparently optimal for tanh.
					low=low,
					high=high,
					size=(self.input_dimension, self.hidden_dimension)),
				dtype=theano.config.floatX),
			name="W",
			borrow=True)


		self.bias = theano.shared(
			value=numpy.zeros((self.hidden_dimension,),
				dtype=theano.config.floatX),
			name="b",
			borrow=True)

		self.reverse_bias = theano.shared(
			value=numpy.zeros((self.input_dimension,),
				dtype=theano.config.floatX),
			name="b'",
			borrow=True)

		self.reverse_weights = self.weights.T	# Tied weights, so the reverse weight
												# matrix is just the transpose.

	def initialise_rng(self):
		"""
		Initialises and subsequently stores a NumPy RandomState.
		"""

		self.rng = numpy.random.RandomState(seed)

	def initialise_symbolic_input(self):
		"""
		Initialises and subsequently stores a symbolic input value.
		"""

		self.symbolic_input = theano.tensor.matrix("x")

	def get_hidden_output(self):
		"""
		Get the values output by the hidden layer.
		"""

		return self.activation(
			theano.tensor.dot(self.symbolic_input, self.weights) +
			self.bias)

	def get_reconstructed_input(self):
		"""
		Get the reconstructed input.
		"""

		return self.activation(
			theano.tensor.dot(self.get_hidden_output(), self.reverse_weights) +
			self.reverse_bias)

	def get_cost(self):
		"""
		Get the symbolic cost.
		"""

		negative_log_loss = -theano.tensor.sum(self.symbolic_input *
			theano.tensor.log(self.get_reconstructed_input()) +
			(1 - self.symbolic_input) *
			theano.tensor.log(1 - self.get_reconstructed_input()),
			axis=1)

		mean_loss = theano.tensor.mean(negative_log_loss)

		return mean_loss

	def get_updates(self):
		"""
		Get a list of updates to make when the model is trained.
		"""

		cost = self.get_cost()
		
		weight_gradient = theano.tensor.grad(cost, self.weights)
		bias_gradient = theano.tensor.grad(cost, self.bias)
		reverse_bias_gradient = theano.tensor.grad(cost, self.reverse_bias)

		updates = [
			(self.weights, self.weights - self.learning_rate*weight_gradient),
			(self.bias, self.bias - self.learning_rate*bias_gradient),
			(self.reverse_bias, self.reverse_bias -
				self.learning_rate*reverse_bias_gradient)]

		return updates

	def initialise_theano_functions(self):
		"""
		Compile Theano functions for symbolic variables.
		"""

		self.train_model_once = theano.function([self.symbolic_input],
			outputs=self.get_cost(),
			updates=self.get_updates())