#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An implementation of RMI with logistic regression.

Principal Author: Buck Shlegeris
"""

from __future__ import division

from itertools import izip

import theano
import numpy

import lib.mnist as mnist
from logistic_regression import Classifier

class RMILogisticClassifier(Classifier):
  """
  Uses RMI and logistic regression to learn to classify data.
  Maybe it will be faster? Who knows.
  """
  def __init__(self, input_batch, output_batch, input_dimension, output_dimension, learning_rate=0.01):
    self.initialize_reward_modulation()
    super(RMILogisticClassifier, self).__init__(input_batch, output_batch, input_dimension,
                                                output_dimension, learning_rate)

  def initialize_reward_modulation(self):
    """
    Initialise the bias vector and store it in a shared Theano value as self.b.
    """

    # Store the bias vector.
    self.modulation = theano.tensor.dscalar('modulation')

  def get_cost(self):
    """
    Return a symbolic linear combination of negative log-likelihood of
    the correct labels according to the model and the total likelihood of
    the input.

    With self.modulation = 1, this is the same as the logistic regression
    classifier.
    """
    supervised_cost = -theano.tensor.mean(
      theano.tensor.log(
        self.get_probability_matrix()
      )[theano.tensor.arange(self.y.shape[0]), self.y]
    ) * self.modulation

    print supervised_cost

    print unsupervised_cost

    return (supervised_cost * self.modulation
            + unsupervised_cost * (1 - self.modulation))

  def initialise_theano_functions(self):
    """
    Set up Theano symbolic functions and store them in self.
    """

    gradient_wrt_W = theano.tensor.grad(cost=self.get_cost(), wrt=self.W)
    gradient_wrt_b = theano.tensor.grad(cost=self.get_cost(), wrt=self.b)
    updates = [
      (self.W, self.W - self.learning_rate * gradient_wrt_W),
      (self.b, self.b - self.learning_rate * gradient_wrt_b)
    ]
    index = theano.tensor.lscalar()
    batch_size = theano.tensor.lscalar()

    self.train_model_once = theano.function(
      inputs=[index, batch_size, self.modulation],
      outputs=self.get_cost(),
      updates=updates,
      givens={
        self.x: self.input_batch[index*batch_size:(index+1)*batch_size],
        self.y: self.output_batch[index*batch_size:(index+1)*batch_size]
      }
    )

  def train_model(self, epochs=100, minibatch_size=600, test_each_epoch=False, test_set=None):
    if test_each_epoch:
      self_accuracies = []
      test_accuracies = []
      test_images, test_labels = test_set

    for epoch in xrange(1, epochs+1):
      batches = self.input_batch.get_value(borrow=True).shape[0]//minibatch_size
      for index in xrange(batches):
        self.train_model_once(index, minibatch_size, 1)
      print "{epoch}/{epochs}: {batch}/{batches}".format(epoch=epoch,
                                                        epochs=epochs,
                                                        batch=index+1,
                                                        batches=batches)
      if test_each_epoch:
        self_accuracies.append(self.calculate_wrongness(self.input_batch, self.output_batch))
        test_accuracies.append(self.calculate_wrongness(test_images, test_labels))

    if test_each_epoch:
      return (self_accuracies, test_accuracies)

if __name__ == '__main__':
  print "loading training images"
  images = mnist.load_training_images(format="theano")
  print "loading training labels"
  labels = mnist.load_training_labels(format="theano")
  print "instantiating classifier"
  classifier = RMILogisticClassifier(images, labels, 28*28, 10)
  print "training...",
  classifier.train_model(10)
  print "done."


  print "loading test images"
  test_images = mnist.load_test_images(format="theano")
  print "loading test labels"
  test_labels = mnist.load_test_labels(format="theano")

  print "Wrong {:.02%} of the time".format(classifier.calculate_wrongness(
                                test_images, test_labels))
  print "(On the training set, wrong {:.02%} of the time)".format(
      classifier.calculate_wrongness(images, labels))