#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classify data using a multilayer perceptron with one hidden layer and RMI.

Principal Author: Matthew Alger
"""

from __future__ import division

import time

import theano
import numpy

import hidden_layer_perceptron

class RMI_Perceptron(hidden_layer_perceptron.Hidden_Layer_Perceptron):
	"""
	Implementation of RMI with a hidden layer perceptron.
	"""

	def __init__(self, *args, **kwargs):
		self.modulation = 0.5
		super(RMI_Perceptron, self).__init__(*args, **kwargs)

		return self.logistic_regression_layer.get_cost()

	def get_cost(self):
		"""
		Get the symbolic cost.
		"""
		something = -theano.tensor.mean(
			theano.tensor.log(self.get_probability_matrix())[
				theano.tensor.arange(self.symbolic_output.shape[0]),
				self.symbolic_output])



		return
			self.regularisation_weights["L1"] * self.regularisation["L1"] +
			self.regularisation_weights["L2"] * self.regularisation["L2"])

	# I probably need to change this?
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

		a bunch of times:
			cost = self.train_model_once(thing, label)



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