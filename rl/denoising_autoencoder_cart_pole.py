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
import cart_pole

class Cart_Pole_DA(Denoising_Autoencoder):
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
		super(Cart_Pole_DA, self).__init__(*args, **kwargs)

	def get_reward(self):
		"""
		Get the symbolic reward.
		"""

		return self.symbolic_output

	def get_lr_cost(self):
		"""
		Get the symbolic cost for the weight matrix and bias vectors.
		"""

		actual_reward = self.get_reward()
		expected_reward = self.get_symbolic_expected_rewards()[
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

if __name__ == '__main__':

	cart = cart_pole.Cart(math.pi/2+0.1)

	input_dimension = len(cart.get_state()) # The length of a state vector
	hidden_dimension = 429 # Arbitrary, at present
	output_dimension = 3 # 3 possible actions

	agent = Cart_Pole_DA(input_dimension, hidden_dimension, output_dimension, gamma=0.9)

	lengths = []

	import cPickle
	import sys

	i = 0
	while True:
		state_info = cart_pole.get_states(agent, cart)
		states, actions, rewards = map(numpy.asarray, zip(*state_info))
		lengths.append(len(state_info))
		print len(state_info)
		print >> sys.stderr, i, len(state_info)
		i += 1


		agent.train_model_once(states, actions, rewards)

		if i%1000 == 0:
			with open("cartDA.pickle", "w") as f:
				cPickle.dump(agent, f)



	# state = cart.get_state()
	# state = numpy.asarray([state])
	# print agent.get_expected_rewards(state)

	def get_action(cart):
		return cart_pole.get_action(agent, cart)

	# cart.reset()
	# cart_pole.animate_cart(cart, get_action)