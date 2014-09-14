#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classify data using a multilayer perceptron with one hidden layer.

Principal Author: Matthew Alger
"""

from __future__ import division

import time

import theano
import numpy

# theano.config.compute_test_value = 'warn'

class Layer(object):
	"""
	Represents an abstract layer with weights and bias.
	"""

	def __init__(self, *args, **kwargs):
		raise NotImplementedError("Cannot instantiate abstract layer.")

	def initialise_weight_matrix(self, zeros=True):
		"""
		Create a symbolic weight matrix and store it in self.W.
		"""

		if zeros:
			matrix = theano.numpy.zeros(
				(self.input_dimension, self.output_dimension),
				dtype=theano.config.floatX)
		else:
			matrix = numpy.asarray(
				self.rng.uniform(
					# This distribution is apparently optimal for tanh.
					# We also know an optimal distribution for sigmoid.
					# So we'll pretend that every function is either tanh, or sigmoid.
					low=-numpy.sqrt(6/(self.input_dimension + self.output_dimension)),
					high=numpy.sqrt(6/(self.input_dimension + self.output_dimension)),
					size=(self.input_dimension, self.output_dimension)),
				dtype=theano.config.floatX)

		self.W = theano.shared(value=matrix, name="W", borrow=True)

	def initialise_bias_vector(self):
		"""
		Create a symbolic bias vector and store it in self.b.
		"""

		vector = numpy.zeros(
			(self.output_dimension,),
			dtype=theano.config.floatX)

		self.b = theano.shared(value=vector, name="b", borrow=True)

class Hidden_Layer(Layer):
	"""
	A hidden layer in a multilayer perceptron.
	"""

	def __init__(self
		, rng
		, symbolic_input
		, input_dimension
		, output_dimension
		, activation=theano.tensor.tanh):
		"""
		rng: NumPy RandomState to initialise weights.
		symbolic_input: Symbolic variable for an input minibatch.
		input_dimension: Dimension of input.
		output_dimension: Number of hidden nodes.
		activation: Nonlinear activation function (default: tanh).
		"""

		self.rng = rng
		self.symbolic_input = symbolic_input
		self.input_dimension = input_dimension
		self.output_dimension = output_dimension
		self.activation = activation

		self.initialise_weight_matrix(zeros=False)
		self.initialise_bias_vector()

	def get_output_matrix(self):
		"""
		Get a symbolic matrix of outputs.
		"""

		return self.activation(
			theano.tensor.dot(self.symbolic_input, self.W) + self.b)

class Logistic_Regression(Layer):
	"""
	Learns to classify and then classifies data by logistic regression.
	"""

	def __init__(self
		, symbolic_input
		, symbolic_output
		, input_dimension
		, output_dimension):
		"""
		symbolic_input: Symbolic variable for an input minibatch.
		??? Is symbolic_output the symbolic variable for an output batch?
		input_dimension: Dimension of input.
		output_dimension: Number of output labels.
		"""

		self.symbolic_input = symbolic_input
		self.symbolic_output = symbolic_output
		self.input_dimension = input_dimension
		self.output_dimension = output_dimension

		self.initialise_weight_matrix()
		self.initialise_bias_vector()

	def get_probability_matrix(self):
		"""
		Get the symbolic probability matrix.
		"""

		return theano.tensor.nnet.softmax(
			theano.tensor.dot(self.symbolic_input, self.W) + self.b)

	def get_most_likely_label(self):
		"""
		Predict what the most likely label is for each input in the minibatch.
		"""

		return theano.tensor.argmax(self.get_probability_matrix(), axis=1)

	def get_cost(self):
		"""
		Get the symbolic cost.
		"""

		return -theano.tensor.mean(
			theano.tensor.log(self.get_probability_matrix())[
				theano.tensor.arange(self.symbolic_output.shape[0]),
				self.symbolic_output])

	def error_rate(self, labels):
		"""
		Calculate the error rate.

		labels: A vector of correct labels for each input in the input minibatch.
		"""

		return theano.tensor.mean(theano.tensor.neq(
			self.get_most_likely_label(),
			labels))

class Hidden_Layer_Perceptron(object):
	"""
	Learns to classify and then classifies data using hidden layers and
	logistic regression.
	"""

	def __init__(self
		, input_batch
		, output_batch
		, validation_input_batch
		, validation_output_batch
		, input_dimension
		, hidden_dimension
		, output_dimension
		, learning_rate=0.01
		, patience=10000
		, patience_boost=2
		, significance_threshold=0.995
		, seed=None
		, regularisation_weights=(0.0, 0.0001)):
		"""
		input_batch: Matrix of all input vectors.
		output_batch: Vector of output labels corresponding to input vectors.
		validation_input_batch: Matrix of all validation input vectors.
		validation_output_batch: Vector of output labels corresponding to
			validation input vectors.
		input_dimension: Dimension of input.
		hidden_dimension: Number of hidden nodes.
		output_dimension: Number of labels in output.
		learning_rate: Gradient descent learning rate.
		patience: How long before early termination can potentially occur.
		patience_boost: How much to boost patience by when a new best is found.
		significance_threshold: How much improvement is considered significant.
		seed: Seed for the hidden layer weights.
		regularisation_weights: Tuple of weights for L1 and L2 regularisation.
		"""

		self.rng = numpy.random.RandomState(seed)
		self.input_batch = input_batch
		self.output_batch = output_batch
		self.validation_input_batch = validation_input_batch
		self.validation_output_batch = validation_output_batch
		self.input_dimension = input_dimension
		self.hidden_dimension = hidden_dimension
		self.output_dimension = output_dimension
		self.learning_rate = learning_rate
		self.patience = patience
		self.patience_boost = patience_boost
		self.significance_threshold = significance_threshold
		self.regularisation_weights = regularisation_weights

		self.initialise_symbolic_input()
		self.initialise_layers()
		self.initialise_regularisation()
		self.initialise_theano_functions()

	def initialise_regularisation(self):
		"""
		Setup L1 and L2 regularisation.
		"""

		self.regularisation = {
			"L1": abs(self.hidden_layer.W).sum() +
				abs(self.logistic_regression_layer.W).sum(),
			"L2": (self.hidden_layer.W ** 2).sum() +
			(self.logistic_regression_layer.W ** 2).sum()}

		self.regularisation_weights = {
			"L1": self.regularisation_weights[0],
			"L2": self.regularisation_weights[1]
		}

	def initialise_layers(self):
		"""
		Setup the hidden and logistic layers.
		"""

		self.hidden_layer = Hidden_Layer(self.rng
			, self.symbolic_input
			, self.input_dimension
			, self.hidden_dimension)

		self.logistic_regression_layer = Logistic_Regression(
			self.hidden_layer.get_output_matrix(),
			self.symbolic_output,
			self.hidden_dimension,
			self.output_dimension)

		self.Ws = [self.hidden_layer.W, self.logistic_regression_layer.W]
		self.bs = [self.hidden_layer.b, self.logistic_regression_layer.b]

	def initialise_symbolic_input(self):
		"""
		Setup input minibatch and classification labels in symbolic form.
		"""

		self.symbolic_input = theano.tensor.matrix("x")
		# self.symbolic_input.tag.test_value = theano.numpy.zeros(
		# 		(600, self.input_dimension),
		# 		dtype=theano.config.floatX)
		self.symbolic_output = theano.tensor.ivector("y")
		# self.symbolic_output.tag.test_value = numpy.zeros(
		# 	(self.input_dimension,),
		# 	dtype=theano.config.floatX)

	def get_cost(self):
		"""
		Get the symbolic cost.
		"""

		return (self.logistic_regression_layer.get_cost() +
			self.regularisation_weights["L1"] * self.regularisation["L1"] +
			self.regularisation_weights["L2"] * self.regularisation["L2"])

	def error_rate(self, labels):
		"""
		Get the error rate of the model.

		labels: A vector of correct labels for each input in the input minibatch.
		"""

		return self.logistic_regression_layer.error_rate(labels)

	def calculate_updates(self):
		"""
		Find gradients of the model parameters and use them to calculate
		the updated parameters.

		Return a list of update tuples.
		"""

		updates = []

		for W in self.Ws:
			gradient = theano.tensor.grad(self.get_cost(), W)
			updates.append((W, W - self.learning_rate * gradient))

		for b in self.bs:
			gradient = theano.tensor.grad(self.get_cost(), b)
			updates.append((b, b - self.learning_rate * gradient))

		return updates

	def initialise_theano_functions(self):
		"""
		Compile Theano functions.
		"""

		index = theano.tensor.lscalar("i")
		batch_size = theano.tensor.lscalar("s")

		self.train_model_once = theano.function(inputs=[index, batch_size]
			, outputs=self.get_cost()
			, updates=self.calculate_updates()
			, givens={
				self.symbolic_input:
					self.input_batch[index*batch_size:(index+1)*batch_size],
				self.symbolic_output:
					self.output_batch[index*batch_size:(index+1)*batch_size]})

		self.validate_model = theano.function(inputs=[index, batch_size]
			, outputs=self.error_rate(self.symbolic_output)
			, givens={
				self.symbolic_input:
					self.validation_input_batch[
						index*batch_size:(index+1)*batch_size],
				self.symbolic_output:
					self.validation_output_batch[
						index*batch_size:(index+1)*batch_size]})

	def train_model(self, epochs=100, minibatch_size=600, yield_every_iteration=False):
		"""
		Train the model against the given data.
		"""

		input_size = self.input_batch.get_value(borrow=True).shape[0]
		batch_count = input_size//minibatch_size
		validation_count = self.validation_input_batch.get_value(
			borrow=True).shape[0]
		validation_batch_count = validation_count//minibatch_size
		validation_frequency = min(batch_count, self.patience/2)

		# Store the best results so far.
		best_W = None
		best_b = None
		best_validation_cost = numpy.inf
		best_iteration = 0

		# Keep track of how long this takes.
		start_time = time.time()
		last_time = start_time

		for epoch in xrange(1, epochs+1):
			for index in xrange(batch_count):
				cost = self.train_model_once(index, minibatch_size)
				iteration = (epoch - 1) * batch_count + index
				if yield_every_iteration:
					yield (iteration, cost)

				if iteration % validation_frequency == 1:
					validation_cost = numpy.asarray([
						self.validate_model(v_index, minibatch_size)
						for v_index in xrange(validation_batch_count)]).mean()
					now_time = time.time() - last_time
					last_time = time.time()
					if not yield_every_iteration:
						yield (epoch, validation_cost, now_time)
					else:
						print "{}: {:.02%}".format(epoch, validation_cost)

					# Update best results.
					if validation_cost < best_validation_cost:
						# Update patience.
						if (validation_cost <
							best_validation_cost * self.significance_threshold):
							patience = max(self.patience, iteration * self.patience_boost)
							# Note that patience only increases if we have done
							# enough iterations.

						best_validation_cost = validation_cost
						best_iteration = iteration

				if self.patience <= iteration:
					break
			else:
				# If we didn't hit the patience threshold, keep looping.
				continue
			break

		total_time = time.time() - start_time
		time_per_epoch = total_time/epochs
		print "Average time per epoch: {}s".format(time_per_epoch)
		raise StopIteration("No more epochs.")

if __name__ == '__main__':
	import lib.mnist as mnist

	print "loading training images"
	images = mnist.load_training_images(format="theano", validation=False)
	validation_images = mnist.load_training_images(format="theano", validation=True)
	print "loading training labels"
	labels = mnist.load_training_labels(format="theano", validation=False)
	validation_labels = mnist.load_training_labels(format="theano", validation=True)
	print "instantiating classifiers"
	classifier = Hidden_Layer_Perceptron(images, labels, validation_images,
		validation_labels, 28*28, 200, 10)
	print "training..."

	import lib.plot as plot
	plot.plot_over_iterators([(i[1] for i in classifier.train_model(5, 600, False))],
		("200",))

	print "done."