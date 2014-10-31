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
		, output_dimension
		, input_batch=None
		, output_batch=None
		, symbolic_input=None
		, rng=None
		, theano_rng=None
		, learning_rate=0.1
		, corruption=0.3):
		"""
		input_dimension: The dimension of the input vectors.
		hidden_dimension: How many hidden nodes to map to.
		input_batch: Optional. Input data.
		output_batch: Optional. A vector of labels corresponding to each input vector.
		output_dimension: How many labels there are.
		symbolic_input: Optional. A symbolic input value.
		rng: Optional. A NumPy RandomState.
		theano_rng: Optional. A Theano RandomStream.
		learning_rate: Optional. How large gradient descent jumps are.
		corruption: Optional. How much to corrupt the input when learning.
		"""

		self.input_dimension = input_dimension
		self.hidden_dimension = hidden_dimension
		self.output_batch = output_batch
		self.output_dimension = output_dimension

		if symbolic_input is None:
			self.initialise_symbolic_input()
		else:
			self.symbolic_input = symbolic_input

		self.initialise_symbolic_output()

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
		reverse bias vector, label weight matrix, and label bias vector.
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

		self.label_weights = theano.shared(
			value=numpy.zeros((self.hidden_dimension, self.output_dimension),
				dtype=theano.config.floatX),
			name="lW",
			borrow=True)

		self.label_bias = theano.shared(
			value=numpy.zeros((self.output_dimension,),
				dtype=theano.config.floatX),
			name="lb",
			borrow=True)

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

	def initialise_symbolic_output(self):
		"""
		Initialises and subsequently stores a symbolic output value.
		"""

		self.symbolic_output = theano.tensor.ivector("y")

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

	def error_rate(self):
		"""
		Get the rate of incorrect prediction.
		"""

		return theano.tensor.mean(theano.tensor.neq(
			self.get_symbolic_predicted_labels(),
			self.symbolic_output))

	def get_cost(self):
		"""
		Get the symbolic cost for the weight matrix and bias vectors.
		"""

		x = self.symbolic_input
		y = self.get_reconstructed_input()

		negative_log_loss = -theano.tensor.sum(x*theano.tensor.log(y) +
			(1-x)*theano.tensor.log(1-y), axis=1)

		mean_loss = theano.tensor.mean(negative_log_loss)

		return mean_loss

	def get_lr_cost(self):
		"""
		Get the symbolic cost for the logistic regression matrix and bias vector.
		"""

		labels = self.get_symbolic_expected_rewards()

		return -theano.tensor.mean(
			theano.tensor.log(labels)[
				theano.tensor.arange(self.symbolic_output.shape[0]),
				self.symbolic_output])

	def get_symbolic_predicted_labels(self):
		"""
		Predict labels of a minibatch.
		"""

		return theano.tensor.argmax(self.get_symbolic_expected_rewards(), axis=1)

	def get_symbolic_expected_rewards(self):
		"""
		Get probabilities of the input values being each label.
		"""

		prob_matrix = theano.tensor.nnet.softmax(
			theano.tensor.dot(self.get_hidden_output(),
				self.label_weights) + self.label_bias)

		return prob_matrix

	def get_updates(self):
		"""
		Get a list of updates to make when the model is trained.
		"""

		da_cost = self.get_cost()

		weight_gradient = theano.tensor.grad(da_cost, self.weights)
		bias_gradient = theano.tensor.grad(da_cost, self.bias)
		reverse_bias_gradient = theano.tensor.grad(da_cost, self.reverse_bias)

		lr_cost = self.get_lr_cost()
		lr_weight_gradient = theano.tensor.grad(lr_cost, self.label_weights)
		lr_bias_gradient = theano.tensor.grad(lr_cost, self.label_bias)

		updates = [
			(self.weights, self.weights - self.learning_rate*weight_gradient),
			(self.bias, self.bias - self.learning_rate*bias_gradient),
			(self.reverse_bias, self.reverse_bias -
				self.learning_rate*reverse_bias_gradient),
			(self.label_weights, self.label_weights -
				self.learning_rate*lr_weight_gradient),
			(self.label_bias, self.label_bias -
				self.learning_rate*lr_bias_gradient)]

		return updates

	def initialise_theano_functions(self):
		"""
		Compile Theano functions for symbolic variables.
		"""

		index = theano.tensor.lscalar("i")
		batch_size = theano.tensor.lscalar("b")
		validation_images = theano.tensor.matrix("vx")
		validation_labels = theano.tensor.ivector("vy")
		input_matrix = theano.tensor.matrix("ix")

		if (self.input_batch is not None and
			self.output_batch is not None):
			self.train_model_once = theano.function([index, batch_size],
				outputs=self.get_cost(),
				updates=self.get_updates(),
				givens={
					self.symbolic_input: self.input_batch[index*batch_size:
						(index+1)*batch_size],
					self.symbolic_output: self.output_batch[index*batch_size:
						(index+1)*batch_size]})

			self.validate_model = theano.function(inputs=[validation_images, validation_labels],
				outputs=self.error_rate(),
				givens={
					self.symbolic_input: validation_images,
					self.symbolic_output: validation_labels},
				allow_input_downcast=True)

		self.get_expected_rewards = theano.function([input_matrix],
			outputs=self.get_symbolic_expected_rewards(),
			givens={self.symbolic_input: input_matrix})

	def get_weight_matrix(self):
		"""
		Get the weight matrix.
		"""

		return self.weights.get_value(borrow=True)

	def train_model(self
		, epochs=100
		, minibatch_size=20
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
		if self.output_batch is None:
			raise ValueError("RMI denoising autoencoder must be initialised "
				"with output data to train model independently.")

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

def test_DA(DA, epochs=15):
	from sys import argv

	import lib.mnist as mnist

	print "loading training images"
	images = mnist.load_training_images(format="theano", validation=False, div=256.0)
	labels = mnist.load_training_labels(format="theano", validation=False)
	print "loading test images"
	validation_images = mnist.load_training_images(format="numpy", validation=True)
	validation_labels = mnist.load_training_labels(format="numpy", validation=True)
	print "instantiating denoising autoencoder"

	corruption = 0.3
	learning_rate = 0.1
	hiddens = 500

	da = DA(784, hiddens,
		input_batch=images,
		output_batch=labels,
		output_dimension=10,
		corruption=corruption,
		learning_rate=learning_rate)
	print "training..."

	for epoch, cost in da.train_model(epochs):
		print epoch, cost
		print "wrong {:.02%} of the time".format(
			float(da.validate_model(validation_images, validation_labels)))

	print "done."

	import PIL
	import lib.dlt_utils as utils
	import random
	image = PIL.Image.fromarray(utils.tile_raster_images(
		X=da.weights.get_value(borrow=True).T,
		img_shape=(28, 28), tile_shape=(50, 10),
		tile_spacing=(1, 1)))
	image.save('../plots/{:010x}_{}_{}_{}_{}_{}.png'.format(
		random.randrange(16**10), argv[0].replace("/", "-"), corruption, learning_rate, epochs, hiddens))

if __name__ == '__main__':
	test_DA(Denoising_Autoencoder, 10)
