#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classify data using a multilayer perceptron with one hidden layer and RMI.

Principal Author: Buck Shlegeris
"""

from __future__ import division

import time

import theano
import numpy

import hidden_layer_perceptron

class RMI_Perceptron(hidden_layer_perceptron.Hidden_Layer_Perceptron):
	"""
	Implementation of RMI with a hidden layer perceptron, to deal with a contextual
	bandit problem.

	In this situation, the 'probability_matrix' means the matrix of expected rewards.
	"""

	def __init__(self, *args, **kwargs):
		self.modulation = 0.5
		super(RMI_Perceptron, self).__init__(*args, **kwargs)

	#	return self.logistic_regression_layer.get_cost()

	def get_cost(self):
		"""
		Get the symbolic cost.
		"""

		prediced_rewards = self.logistic_regression_layer.get_probability_matrix()[
				theano.tensor.arange(self.symbolic_output.shape[0]),
				self.get_actions()]

		actual_rewards = theano.tensor.eq(self.symbolic_output, self.get_actions())

		wrongness = -theano.tensor.mean(prediced_rewards - actual_rewards)

		return (wrongness +
			self.regularisation_weights["L1"] * self.regularisation["L1"] +
			self.regularisation_weights["L2"] * self.regularisation["L2"])


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


	def get_actions(self):
		"""
		Predict what the most likely label is for each input in the minibatch.
		"""

		# for the moment, let's forget exploration
		bests = theano.tensor.argmax(self.logistic_regression_layer.get_probability_matrix(), axis=1)
		return bests

if __name__ == '__main__':
	import lib.mnist as mnist

	print "loading training images"
	images = mnist.load_training_images(format="theano", validation=False)
	validation_images = mnist.load_training_images(format="theano", validation=True)
	print "loading training labels"
	labels = mnist.load_training_labels(format="theano", validation=False)
	validation_labels = mnist.load_training_labels(format="theano", validation=True)
	print "instantiating classifiers"
	rmi_classifier = RMI_Perceptron(images, labels, validation_images,
		validation_labels, 28*28, 500, 10)
	hlp_classifier = hidden_layer_perceptron.Hidden_Layer_Perceptron(images,
		labels, validation_images,
		validation_labels, 28*28, 500, 10)
	print "training..."

	import lib.plot as plot
	plot.plot_over_iterators([(i[1] for i in rmi_classifier.train_model(100, 600, True)),
		(i[1] for i in hlp_classifier.train_model(100, 600, True))], ("rmi", "hlp"))

	print "done."