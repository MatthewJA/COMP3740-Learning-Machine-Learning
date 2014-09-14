#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Learns features of inputs with reward-modulated inference.

Principal Author: Matthew Alger
"""

from __future__ import division

import numpy
import theano

from denoising_autoencoder import Denoising_Autoencoder, test_DA

class RMI_DA(Denoising_Autoencoder):
	"""
	Denoising autoencoder that also considers expected reward while
	learning.
	"""

	def __init__(self, *args, **kwargs):
		"""
		Same as Denoising_Autoencoder, but with
		an additional keyword argument:

		modulation: Percentage weighting to give labels.
		"""

		self.modulation = kwargs.pop("modulation", 0)
		super(RMI_DA, self).__init__(*args, **kwargs)

	def get_cost(self):
		"""
		Get the symbolic cost for the weight matrix and bias vectors.
		"""

		da_loss = super(RMI_DA, self).get_cost()

		label_loss = self.get_lr_cost()

		return da_loss * (1 - self.modulation) + label_loss * self.modulation

	def train_model(self, *args, **kwargs):
		"""
		Train the model against the given data.
		"""

		iterator = super(RMI_DA, self).train_model(*args, **kwargs)
		for i in iterator:
			yield i
			self.modulation = self.modulation + (1-self.modulation)/5
			print "modulation is now {}".format(self.modulation)

if __name__ == '__main__':
	test_DA(RMI_DA)