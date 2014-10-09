"""
Cart-pole problem with a denoising autoencoder.

Principal Author: Matthew Alger
"""

from __future__ import division

import math

import numpy
import theano
import pylab

from denoising_autoencoder import Denoising_Autoencoder, test_DA

class MDP_DA(Denoising_Autoencoder):
	"""
	Denoising autoencoder implementation of the cart-pole problem.
	"""

	def __init__(self, *args, **kwargs):
		"""
		Same as denoising autoencoder, but with an additional keyword argument

		gamma: Reward discount rate.
		"""

		self.gamma = kwargs.pop("gamma", 0.9)
		self.action_vector = theano.tensor.ivector("av")
		super(MDP_DA, self).__init__(*args, **kwargs)

	def get_reward(self):
		"""
		Get the symbolic reward.
		"""

		return self.symbolic_output

	def get_symbolic_expected_actual_rewards(self):
		"""
		Get expected rewards (not probabilities!)
		Note that this works for negative reward values too.
		"""

		prob_matrix = theano.tensor.nnet.softmax(
			theano.tensor.dot(self.get_hidden_output(),
				self.label_weights) + self.label_bias)

		return prob_matrix

	def get_lr_cost(self):
		"""
		Get the symbolic cost for the weight matrix and bias vectors.
		"""

		actual_reward = self.get_reward()
		expected_reward = self.get_symbolic_expected_actual_rewards()[
			theano.tensor.arange(self.action_vector.shape[0]),
				self.action_vector]

		reward_difference = theano.tensor.mean(
			abs(actual_reward - expected_reward))
		return reward_difference

	def initialise_symbolic_output(self):
		"""
		Initialises and subsequently stores a symbolic output value.
		"""

		self.symbolic_output = theano.tensor.dvector("y")

	def initialise_theano_functions(self):
		"""
		Compile Theano functions for symbolic variables.
		"""

		input_matrix = theano.tensor.matrix("ix")
		output_vector = theano.tensor.dvector("ox")
		action_vector = theano.tensor.ivector("av")

		self.train_model_once = theano.function(
			[input_matrix, action_vector, output_vector],
			outputs=self.get_cost(),
			updates=self.get_updates(),
			givens={
				self.symbolic_input: input_matrix,
				self.symbolic_output: output_vector,
				self.action_vector: action_vector},
			allow_input_downcast=True)

		self.get_expected_rewards = theano.function([input_matrix],
			outputs=self.get_symbolic_expected_rewards(),
			givens={self.symbolic_input: input_matrix})

	def train_model(self, *args, **kwargs):
		raise NotImplementedError("Cart Pole DA can't self-train.")

