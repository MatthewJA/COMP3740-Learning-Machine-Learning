#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classify data using a multilayer perceptron with one hidden layer.
"""

from __future__ import division

import theano
import numpy

import mnist

class Hidden_Layer(object):
	"""
	Represents a hidden layer in a multilayer perceptron.
	"""

	def __init__(self
		, rng
		, input_dimension
		, output_dimension
		, activation=theano.tensor.tanh):
		"""
		rng: NumPy RandomState to initialise weights.
		input_dimension: Dimension of input vectors.
		output_dimension: Number of hidden nodes.
		activation: A nonlinear activation function.
		"""

		self.rng = rng
		self.input_dimension = input_dimension
		self.output_dimension = output_dimension
		self.activation = activation

		self.initialise_symbolic_input()
		self.initialise_weight_matrix()
		self.initialise_bias_vector()

	def initialise_weight_matrix(self):
		"""
		Initialise the weight matrix and store it in a shared Theano value as self.W.
		"""

		matrix = numpy.asarray(
			self.rng.uniform(
				low=-numpy.sqrt(6/(self.input_dimension + self.output_dimension)),
				high=numpy.sqrt(6/(self.input_dimension + self.output_dimension)),
				size=(self.input_dimension, self.output_dimension)
			),
			dtype=theano.config.floatX
		)

		if activation == theano.tensor.nnet.sigmoid:
			# Sigmoid functions have a wider range of initial values.
			# Other functions probably do too but I don't know what those
			# values ought to be, so let's pretend that every activation
			# function is actually just tanh and the only one that isn't
			# is sigmoid.
			matrix *= 4

		self.W = theano.shared(
			value=matrix,
			name="W"
		)

	def initialise_bias_vector(self):
		"""
		Initialise the bias vector and store it in a shared Theano value as self.W.
		"""

		# Initialise a bias vector b.
		vector = theano.numpy.zeros(
			(self.output_dimension,),
			dtype=theano.config.floatX
		)

		# Store the bias vector.
		self.b = theano.shared(
			value=vector,
			name="b"
		)

	def initialise_symbolic_input(self):
		"""
		Initialise the symbolic version of the input.
		"""

		self.x = theano.tensor.matrix()

	def get_probability_matrix(self):
		"""
		Get symbolic probability matrix.
		"""

		return self.activation(theano.dot(self.x, self.W) + self.b)

class Hidden_Layer_Perceptron(object):
	"""
	Learns to classify and then classifies data.
	"""

	def __init__(self
		, input_batch
		, output_batch
		, input_dimension
		, output_dimension
		, hidden_layer_dimension=500
		, learning_rate=0.01):
		"""
		input_batch: NumPy array of input vectors.
		output_batch: NumPy array of output labels.
		input_dimension: Dimension of input vectors.
		output_dimension: Number of output labels.
		hidden_layer_dimension: Dimension of hidden layer.
		learning_rate: Size of jumps to make while learning.
		"""

		self.input_batch = input_batch
		self.output_batch = output_batch
		self.input_dimension = input_dimension
		self.output_dimension = output_dimension
		self.hidden_layer_dimension = 500
		self.learning_rate = learning_rate