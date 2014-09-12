#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Learns features of inputs with reward-modulated inference.

Principal Author: Matthew Alger
"""

from __future__ import division

import numpy
import theano

from denoising_autoencoder import Denoising_Autoencoder

class RMI_DA(Denoising_Autoencoder):
	"""
	It like learns how to take noisy versions of the input to unnoisy versions
	of the input like back to the original versions of the input. -- Buck
	"""

	def __init__(self, *args, **kwargs):
		"""
		Same as Denoising_Autoencoder, but with additional keyword arguments:

		output_batch: A vector of labels corresponding to each input vector.
		output_dimension: How many labels there are.
		modulation: Percentage weighting to give labels.
		"""

		self.output_batch = kwargs.pop("output_batch", None)
		self.modulation = kwargs.pop("modulation", 0)
		self.output_dimension = kwargs.pop("output_dimension")
		super(RMI_DA, self).__init__(*args, **kwargs)

	def initialise_symbolic_input(self):
		"""
		Initialises and subsequently stores a symbolic input value.
		"""

		self.symbolic_output = theano.tensor.ivector("y")
		super(RMI_DA, self).initialise_symbolic_input()

	def get_predictions(self):
		"""
		Get predictions of what input values we read in.
		"""

		prob_matrix = theano.tensor.nnet.softmax(
			theano.tensor.dot(self.get_hidden_output(),
				self.label_weights) + self.label_bias)

		return prob_matrix

	def make_prediction(self):
		"""
		Predict labels of a minibatch.
		"""

		return theano.tensor.argmax(self.get_predictions(), axis=1)

	def initialise_parameters(self):
		"""
		Initialises and subsequently stores a weight matrix and bias
		vector for label prediction, and also all the regular denoising
		autoencoder parameters.
		"""

		self.label_weights = theano.shared(
			value=numpy.zeros((self.hidden_dimension, self.input_dimension),
				dtype=theano.config.floatX),
			name="lW",
			borrow=True)

		self.label_bias = theano.shared(
			value=numpy.zeros((self.input_dimension,),
				dtype=theano.config.floatX),
			name="lb",
			borrow=True)

		super(RMI_DA, self).initialise_parameters()

	def get_cost(self):
		"""
		Get the symbolic cost for the weight matrix and bias vectors.
		"""

		x = self.symbolic_input
		y = self.get_reconstructed_input()

		negative_log_loss = -theano.tensor.sum(x*theano.tensor.log(y) +
			(1-x)*theano.tensor.log(1-y), axis=1)

		mean_nll = theano.tensor.mean(negative_log_loss)

		label_loss = self.get_lr_cost()

		return mean_nll * (1 - self.modulation) + label_loss * self.modulation

	def get_lr_cost(self):
		"""
		Get the symbolic cost for the logistic regression matrix and bias vector.
		"""

		labels = self.get_predictions()

		return -theano.tensor.mean(
			theano.tensor.log(labels)[
				theano.tensor.arange(self.symbolic_output.shape[0]),
				self.symbolic_output])

	def error_rate(self):
		"""
		Get the rate of incorrect prediction.
		"""

		return theano.tensor.mean(theano.tensor.neq(
			self.make_prediction(),
			self.symbolic_output))

	def initialise_theano_functions(self):
		"""
		Compile Theano functions for symbolic variables.
		"""

		index = theano.tensor.lscalar("i")
		batch_size = theano.tensor.lscalar("b")
		validation_images = theano.tensor.matrix("vx")
		validation_labels = theano.tensor.ivector("vy")

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

	def get_updates(self):
		"""
		Get a list of updates to make when the model is trained.
		"""

		print "fetching parent updates..."
		updates = super(RMI_DA, self).get_updates()
		print "fetching RMI updates..."

		cost = self.get_lr_cost()
		
		weight_gradient = theano.tensor.grad(cost, self.label_weights)
		bias_gradient = theano.tensor.grad(cost, self.label_bias)

		updates += [
			(self.label_weights, self.label_weights -
				self.learning_rate*weight_gradient),
			(self.label_bias, self.label_bias -
				self.learning_rate*bias_gradient)]

		return updates

	def train_model(self, *args, **kwargs):
		"""
		Train the model against the given data.
		"""

		if self.output_batch is None:
			raise ValueError("RMI denoising autoencoder must be initialised "
				"with output data to train model independently.")

		iterator = super(RMI_DA, self).train_model(*args, **kwargs)
		for i in iterator:
			yield i
			self.modulation = self.modulation + (1-self.modulation)/5
			print "modulation is now {}".format(self.modulation)

if __name__ == '__main__':
	import lib.mnist as mnist

	print "loading training images"
	images = mnist.load_training_images(format="theano", validation=False, div=256.0)
	labels = mnist.load_training_labels(format="theano", validation=False)
	print "instantiating denoising autoencoder"

	corruption = 0.3
	learning_rate = 0.1
	epochs = 15
	hiddens = 500

	da = RMI_DA(784, hiddens, images,
		output_batch=labels,
		corruption=corruption,
		output_dimension=10,
		learning_rate=learning_rate,
		modulation=0)
	print "training..."

	# import lib.plot as plot
	# plot.plot_over_iterators([(i[1]/1000.0 for i in da.train_model(
		# yield_every_iteration=True, epochs=10))], ("dA",))
	for epoch, cost in da.train_model(epochs):
		print epoch, cost

	print "done."

	print "loading test images"
	validation_images = mnist.load_training_images(format="numpy", validation=True)
	validation_labels = mnist.load_training_labels(format="numpy", validation=True)
	print da.validate_model(validation_images, validation_labels)

	# import lib.matrix_viewer as mv
	# mv.view_real_images(da.get_weight_matrix())
	import PIL
	import lib.dlt_utils as utils
	import random
	image = PIL.Image.fromarray(utils.tile_raster_images(
		X=da.weights.get_value(borrow=True).T,
		img_shape=(28, 28), tile_shape=(hiddens//10, 10),
		tile_spacing=(1, 1)))
	image.save('../plots/RMI_DA_{:010x}_{}_{}_{}_{}.png'.format(
		random.randrange(16**10), corruption, learning_rate, epochs, hiddens))