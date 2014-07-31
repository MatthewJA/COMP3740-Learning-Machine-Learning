#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classify data using logistic regression.
"""

from __future__ import division

from itertools import izip

import theano
import numpy

import mnist

class Classifier(object):
	"""
	Learns to classify and then classifies data.

	The classifier will hold a weight matrix W and a bias vector b that it
	uses to classify data. The matrix and the vector comprise the model, and
	can be thought of as the ``knowledge'' of the classfier. Through gradient
	descent the values of W and b will be found and tuned to minimise
	classification error (loss).
	"""

	def __init__(self, input_dimension, output_dimension, learning_rate=0.01):
		"""
		The input dimension is a number representing the dimensions of the input
		vectors. For example, a 28×28 image would be represented by a
		784-dimensional vector, so this parameter would be 784.

		The output dimension is the number of labels that the classifier will
		classify. For example, if you want to detect digits, you would have
		labels {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, which is a 10-dimensional vector,
		so this parameter would be 10.
		"""

		self.input_dimension = input_dimension
		self.output_dimension = output_dimension
		self.learning_rate = learning_rate

		# Initialise parameters.
		self.initialise_weight_matrix()
		self.initialise_bias_vector()
		self.initialise_symbolic_input()

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

	def initialise_symbolic_input(self):
		"""
		Initialise the symbolic version of the input.
		"""

		self.x = theano.tensor.matrix("x")
		self.y = theano.tensor.ivector("y")

	def get_probability_matrix(self):
		"""
		Get symbolic probability matrix.
		"""

		return theano.tensor.nnet.softmax(theano.dot(self.x, self.W) + self.b)

	def get_most_likely_label(self):
		"""
		Get symbolic most likely label.
		"""

		return theano.tensor.argmax(self.get_probability_matrix(), axis=1)

	def make_prediction(self, input_minibatch, input_labels):
		"""
		Predict the label of the items in the minibatch.
		"""

		return theano.function(
			inputs=[],
			outputs=self.get_most_likely_label(),
			givens={
				self.x: numpy.array(input_minibatch, dtype=theano.config.floatX),
				self.y: numpy.array(input_labels)
			},
			on_unused_input="warn"
		)()

	def calculate_wrongness(self, input_minibatch, input_labels):
		"""
		Calculate the wrongness when classifying inputs.
		"""

		predictions = self.make_prediction(input_minibatch, input_labels)
		total = 0
		right = 0
		for prediction, label in izip(predictions, input_labels):
			if prediction == label:
				right += 1
			total += 1

		return 1-right/total

	def get_negative_log_likelihood(self):
		return -theano.tensor.mean(
			theano.tensor.log(
				self.get_probability_matrix()
			)[theano.tensor.arange(self.y.shape[0]), self.y]
		)

	def train_model_once(self, input_minibatch, input_labels):
		gradient_wrt_W = theano.tensor.grad(cost=self.get_negative_log_likelihood(), wrt=classifier.W)
		gradient_wrt_b = theano.tensor.grad(cost=self.get_negative_log_likelihood(), wrt=classifier.b)
		updates = [
			(self.W, self.W - self.learning_rate * gradient_wrt_W),
			(self.b, self.b - self.learning_rate * gradient_wrt_b)
		]
		return theano.function(
			inputs=[],
			outputs=self.get_negative_log_likelihood(),
			updates=updates,
			givens={
				self.x: numpy.array(input_minibatch, dtype=theano.config.floatX),
				self.y: numpy.array(input_labels, dtype="int32")
			},
			on_unused_input="ignore"
		)()

	def train_model(self, input_batch, input_labels, epochs=3, minibatch_size=600):
		for i in xrange(epochs):
			batch_start = 0
			while batch_start < len(input_batch):
				print "{}/{}: batch {}/{}".format(i, epochs, batch_start//minibatch_size+1,len(input_batch)//minibatch_size)
				this_minibatch_size = min(len(input_batch)-batch_start, minibatch_size)
				minibatch = input_batch[batch_start:batch_start+this_minibatch_size]
				labelbatch = input_labels[batch_start:batch_start+this_minibatch_size]
				print "NLL:", self.train_model_once(minibatch, labelbatch)
				batch_start += minibatch_size

if __name__ == "__main__":
	print "loading training images"
	images = mnist.load_training_images()
	print "loading training labels"
	labels = mnist.load_training_labels()
	print "instantiating classifier"
	classifier = Classifier(28*28, 10)
	print "training"
	classifier.train_model(images, labels)
	print "Wrong {:.02%} of the time".format(classifier.calculate_wrongness(images, labels))
