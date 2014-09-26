#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A denoising autoencoder implementation of the contextual bandit problem,
using reward modulation.

Principal Author: Matthew Alger
"""

from __future__ import division

import numpy
import theano

from denoising_autoencoder import test_DA
from denoising_autoencoder_rmi import RMI_DA

class CB_DA(RMI_DA):
	"""
	Denoising autoencoder implementation of the contextual bandit problem,
	using reward modulation.
	"""

	def get_reward(self):
		"""
		Get the symbolic reward.
		"""
		predictions = self.make_prediction()
		results = self.symbolic_output

		reward = theano.tensor.eq(results, predictions)

		return reward

	def get_lr_cost(self):
		"""
		Get the symbolic cost for the weight matrix and bias vectors.
		"""

		reward = self.get_reward()
		expected_reward = self.get_predictions()[
				theano.tensor.arange(self.symbolic_output.shape[0]),
				self.make_prediction()]

		reward_difference = theano.tensor.mean(abs(reward - expected_reward))
		return reward_difference

if __name__ == '__main__':
	test_DA(CB_DA, 25)