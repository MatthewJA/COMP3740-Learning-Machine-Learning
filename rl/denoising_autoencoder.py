#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Learns features of inputs.

Principal Author: Matthew Alger
"""

from __future__ import division

import time

import numpy
import theano
from theano.tensor.shared_randomstreams import RandomStreams

class Denoising_Autoencoder(object):
	"""
	It like learns how to take noisy versions of the input to unnoisy versions
	of the input like back to the original versions of the input. -- Buck
	"""

	def __init__(self
		, input_dimension
		, hidden_dimension
		, input_batch=None
		, symbolic_input=None
		, rng=None
		, theano_rng=None
		, learning_rate=0.1
		, corruption=0.3):
		"""
		input_dimension: The dimension of the input vectors.
		hidden_dimension: How many hidden nodes to map to.
		input_batch: Optional. 
		symbolic_input: Optional. A symbolic input value.
		rng: Optional. A NumPy RandomState.
		theano_rng: Optional. A Theano RandomStream.
		learning_rate: Optional. How large gradient descent jumps are.
		corruption: Optional. How much to corrupt the input when learning.
		"""

		self.input_dimension = input_dimension
		self.hidden_dimension = hidden_dimension

		if symbolic_input is None:
			self.initialise_symbolic_input()
		else:
			self.symbolic_input = symbolic_input

		if rng is None:
			self.initialise_rng()
		else:
			self.rng = rng

		if theano_rng is None:
			self.initialise_theano_rng()
		else:
			self.theano_rng = theano_rng

		self.corruption = corruption
		self.input_batch = input_batch
		self.activation = theano.tensor.nnet.sigmoid
		self.learning_rate = learning_rate
		self.initialise_corrupted_input()
		self.initialise_parameters()
		self.initialise_theano_functions()

	def initialise_corrupted_input(self):
		self.symbolic_corrupted_input = self.theano_rng.binomial(
				size=self.symbolic_input.shape,
				n=1,
				p=1 - self.corruption,
				dtype=theano.config.floatX) * self.symbolic_input

	def initialise_theano_rng(self):
		"""
		Initialise and store a Theano RandomStream.
		"""

		self.theano_rng = RandomStreams(self.rng.randint(2**30))

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

		self.rng = numpy.random.RandomState()

	def initialise_symbolic_input(self):
		"""
		Initialises and subsequently stores a symbolic input value.
		"""

		self.symbolic_input = theano.tensor.dmatrix("x")

	def get_hidden_output(self):
		"""
		Get the values output by the hidden layer.
		"""

		return self.activation(
			theano.tensor.dot(self.symbolic_corrupted_input, self.weights) +
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

		x = self.symbolic_input
		y = self.get_reconstructed_input()

		negative_log_loss = -theano.tensor.sum(x*theano.tensor.log(y) +
			(1-x)*theano.tensor.log(1-y), axis=1)
		# negative_log_loss = -theano.tensor.sum(x*theano.tensor.log(y) + (1-x)*theano.tensor.log(1-y), axis=1)

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

		index = theano.tensor.lscalar("i")
		batch_size = theano.tensor.lscalar("b")

		if self.input_batch is not None:
			self.train_model_once = theano.function([index, batch_size],
				outputs=self.get_cost(),
				updates=self.get_updates(),
				givens={
					self.symbolic_input: self.input_batch[index*batch_size:
						(index+1)*batch_size]
				})

	def get_weight_matrix(self):
		"""
		Get the weight matrix.
		"""

		return self.weights.get_value(borrow=True)

	def train_model(self
		, epochs=100
		, minibatch_size=600
		, yield_every_iteration=False):
		"""
		Train the model against the given data.

		epochs: How long to train for.
		minibatch_size: How large each minibatch is.
		yield_every_iteration: When to yield.
		"""

		if self.input_batch is None:
			raise ValueError("Denoising autoencoder must be initialised with "
				"input data to train model independently.")

		batch_count = self.input_batch.get_value(
			borrow=True).shape[0]//minibatch_size

		for epoch in xrange(epochs):
			costs = []
			for index in xrange(batch_count):
				cost = self.train_model_once(index, minibatch_size)
				costs.append(cost)
				if yield_every_iteration:
					yield (index, cost)

			if not yield_every_iteration:
				yield (epoch, numpy.mean(costs))

if __name__ == '__main__':
	import lib.mnist as mnist

	print "loading training images"
	images = mnist.load_training_images(format="theano", validation=False, div=256.0)
	print "instantiating denoising autoencoder"

	corruption = 0.5
	learning_rate = 0.1

	da = Denoising_Autoencoder(784, 500, images,
		corruption=corruption,
		learning_rate=learning_rate)
	print "training..."

	# import lib.plot as plot
	# plot.plot_over_iterators([(i[1]/1000.0 for i in da.train_model(
		# yield_every_iteration=True, epochs=10))], ("dA",))
	for epoch, cost in da.train_model(15):
		print epoch, cost

	print "done."

	# import lib.matrix_viewer as mv
	# mv.view_real_images(da.get_weight_matrix())
	import PIL
	import lib.dlt_utils as utils
	import random
	image = PIL.Image.fromarray(utils.tile_raster_images(
		X=da.weights.get_value(borrow=True).T,
		img_shape=(28, 28), tile_shape=(10, 10),
		tile_spacing=(1, 1)))
	image.save('../plots/{:010x}_{}_{}.png'.format(
		random.randrange(16**10), corruption, learning_rate))