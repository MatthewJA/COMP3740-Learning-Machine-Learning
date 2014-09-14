#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Change the modulation linearly over epochs in a RMI denoising autoencoder.
"""

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

epochs = 15
minibatch_size = 20
modulation = lambda e: e*1.0/epochs # Linearly increase from 0 to 1.

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
	modulation=modulation,
	change_modulation=change_modulation)

plot.plot_over_iterators(
	[(da.validate_model(validation_inp, validation_out)
		for i in da.train_model(epochs, minibatch_size, False)),
	 (rmi_da.validate_model(validation_inp, validation_out)
		for i in rmi_da.train_model(epochs, minibatch_size, False))],
	("DA", "RMI DA"),
	scale=10)