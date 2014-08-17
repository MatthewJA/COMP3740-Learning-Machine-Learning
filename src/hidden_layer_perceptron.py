#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classify data using a multilayer perceptron with one hidden layer.
"""

from __future__ import division

import theano
import numpy

import mnist
import logistic_regression

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
				size=(self.input_dimension, self.output_dimension)),
			dtype=theano.config.floatX)

		if self.activation == theano.tensor.nnet.sigmoid:
			# Sigmoid functions have a wider range of initial values.
			# Other functions probably do too but I don't know what those
			# values ought to be, so let's pretend that every activation
			# function is actually just tanh and the only one that isn't
			# is sigmoid.
			matrix *= 4

		self.W = theano.shared(
			value=matrix,
			name="hl_W")

	def initialise_bias_vector(self):
		"""
		Initialise the bias vector and store it in a shared Theano value as self.W.
		"""

		# Initialise a bias vector b.
		vector = theano.numpy.zeros(
			(self.output_dimension,),
			dtype=theano.config.floatX)

		# Store the bias vector.
		self.b = theano.shared(
			value=vector,
			name="hl_b")

	def initialise_symbolic_input(self):
		"""
		Initialise the symbolic version of the input.
		"""

		self.x = theano.tensor.matrix("hl_x")

	def get_output_matrix(self):
		"""
		Get symbolic output matrix.

		Similar to the probability matrix in logistic regression.
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
		, learning_rate=0.01
		, seed=None
		, regularisation_weights=(0.01, 0.01)):
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
		self.regularisation_weights = {
			"L1": regularisation_weights[0],
			"L2": regularisation_weights[1]
		}

		self.rng = numpy.random.RandomState(seed)

		self.hidden_layer = Hidden_Layer(self.rng, self.input_dimension,
			self.hidden_layer_dimension)

		self.logistic_regression_layer = logistic_regression.Classifier(
			input_batch=self.hidden_layer.get_output_matrix(),
			output_batch=self.output_batch,
			input_dimension=self.hidden_layer_dimension,
			output_dimension=self.output_dimension,
			learning_rate=self.learning_rate,
			symbolic_input=True)

		self.initialise_norms()
		self.initialise_theano_functions()

	def get_cost(self):
		"""
		Return the symbolic cost.
		"""

		return (self.logistic_regression_layer.get_cost() +
			self.regularisation_weights["L1"] * self.norms["L1"] +
			self.regularisation_weights["L2"] * self.norms["L2sq"])

	def calculate_wrongness(self):
		"""
		Calculate the wrongness when classifying inputs.
		"""

		return self.logistic_regression_layer.calculate_wrongness()

	def initialise_norms(self):
		"""
		Compute and store L1 and L2 norms.
		"""

		self.norms = {
			"L1": abs(self.hidden_layer.W).sum() +
				abs(self.logistic_regression_layer.W).sum(),
			"L2sq": (self.hidden_layer.W ** 2).sum() +
				abs(self.logistic_regression_layer.W ** 2).sum()
		}

	def initialise_theano_functions(self):
		"""
		Set up Theano symbolic functions and store them in self.
		"""

		gradient_wrt_hl_W = theano.tensor.grad(
			cost=self.get_cost(),
			wrt=self.hidden_layer.W)
		gradient_wrt_hl_b = theano.tensor.grad(
			cost=self.get_cost(),
			wrt=self.hidden_layer.b)
		gradient_wrt_lrl_W = theano.tensor.grad(
			cost=self.get_cost(),
			wrt=self.logistic_regression_layer.b)
		gradient_wrt_lrl_b = theano.tensor.grad(
			cost=self.get_cost(),
			wrt=self.logistic_regression_layer.b)

		updates = [
			(self.hidden_layer.W,
				self.hidden_layer.W -
				self.learning_rate * gradient_wrt_hl_W),
			(self.hidden_layer.b,
				self.hidden_layer.b -
				self.learning_rate * gradient_wrt_hl_b),
			(self.logistic_regression_layer.W,
				self.logistic_regression_layer.W -
				self.learning_rate * gradient_wrt_lrl_W),
			(self.logistic_regression_layer.b,
				self.logistic_regression_layer.b -
				self.learning_rate * gradient_wrt_lrl_b)]

		index = theano.tensor.lscalar()
		batch_size = theano.tensor.lscalar()

		self.train_model_once = theano.function(
			inputs=[index, batch_size],
			outputs=self.get_cost(),
			updates=updates,
			givens={
				self.hidden_layer.x: self.input_batch[index*batch_size:(index+1)*batch_size],
				self.logistic_regression_layer.y: self.output_batch[index*batch_size:(index+1)*batch_size]
			}
		)

	def train_model(self, epochs=100, minibatch_size=600, test_each_epoch=False, test_set=None):
		if test_each_epoch:
			self_accuracies = []
			test_accuracies = []
			test_images, test_labels = test_set

		for epoch in xrange(1, epochs+1):
			batches = self.input_batch.get_value(borrow=True).shape[0]//minibatch_size
			for index in xrange(batches):
				self.train_model_once(index, minibatch_size)
			print "{epoch}/{epochs}: {batch}/{batches}".format(
				epoch=epoch,
				epochs=epochs,
				batch=index+1,
				batches=batches)
			if test_each_epoch:
				self_accuracies.append(self.calculate_wrongness(self.input_batch, self.output_batch))
				test_accuracies.append(self.calculate_wrongness(test_images, test_labels))

		if test_each_epoch:
			return (self_accuracies, test_accuracies)

if __name__ == '__main__':
	print "loading training images"
	images = mnist.load_training_images(format="theano")
	print "loading training labels"
	labels = mnist.load_training_labels(format="theano")
	print "instantiating classifier"
	classifier = Hidden_Layer_Perceptron(images, labels, 28*28, 10)
	print "training...",
	classifier.train_model(100)
	print "done."


	print "loading test images"
	test_images = mnist.load_test_images(format="theano")
	print "loading test labels"
	test_labels = mnist.load_test_labels(format="theano")

	print "Wrong {:.02%} of the time".format(classifier.calculate_wrongness(
		test_images, test_labels))
	print "(On the training set, wrong {:.02%} of the time)".format(
		classifier.calculate_wrongness(images, labels))