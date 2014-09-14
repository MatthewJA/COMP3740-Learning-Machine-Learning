#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Change the modulation with sigmoid over epochs in a RMI denoising autoencoder.

Principal Author: Matthew Alger
"""

import numpy

from denoising_autoencoder_rmi import RMI_DA
from denoising_autoencoder import Denoising_Autoencoder as DA
import lib.mnist as mnist
import lib.plot as plot

print "Loading data."
inp = mnist.load_training_images(
	format="theano",
	validation=False,
	div=256.0)
out = mnist.load_training_labels(
	format="theano",
	validation=False)
validation_inp = mnist.load_training_images(
	format="numpy",
	validation=True)
validation_out = mnist.load_training_labels(
	format="numpy",
	validation=True)

corruption = 0.3
learning_rate = 0.1
hiddens = 500

epochs = 30
minibatch_size = 20
change_modulation = lambda e: 1.0/(1 + numpy.exp(epochs/2.0 - e)) # Sigmoidally increase from 0 to 1.

da = DA(
	784,
	hiddens,
	10,
	inp,
	out,
	learning_rate=learning_rate,
	corruption=corruption)

rmi_da = RMI_DA(
	784,
	hiddens,
	10,
	inp,
	out,
	learning_rate=learning_rate,
	corruption=corruption,
	modulation=change_modulation)

def da_validate(a=[0]):
	a[0] += 1 # NEVER DO THIS
	print "epoch {}".format(a[0])
	return da.validate_model(validation_inp, validation_out)

def rmi_da_validate(a=[0]):
	a[0] += 1 # EVER
	print "epoch {}".format(a[0])
	return rmi_da.validate_model(validation_inp, validation_out)

plot.plot_over_iterators(
	[(da_validate()
		for i in da.train_model(epochs, minibatch_size, False)),
	 (rmi_da_validate()
		for i in rmi_da.train_model(epochs, minibatch_size, False))],
	("DA", "RMI DA"),
	scale=10)