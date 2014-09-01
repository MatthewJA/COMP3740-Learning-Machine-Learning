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

		return (self.modulation * self.get_supervised_cost() +
			(1-self.modulation) * self.get_unsupervised_cost() +
			self.regularisation_weights["L1"] * self.regularisation["L1"] +
			self.regularisation_weights["L2"] * self.regularisation["L2"])

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