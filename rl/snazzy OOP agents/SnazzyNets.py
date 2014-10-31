import numpy
import math
import theano
import theano.tensor as T


theano.config.exception_verbosity = "high"

from SnazzyLayers import *

class SnazzyNet(object):
  def __init__(self):
    assert self.cost
    assert self.input
    self.compile_functions()

  def compile_functions(self):
    self.predictor.compile_learning_function(self)

    self.regression_function = theano.function(
      inputs=[self.input.output],
      outputs=self.output_layer.output)

  def update(self, layer, learning_rate):
    layer.update(self.cost, learning_rate)

  def get_regression(self, input_vector):
    assert input_vector.shape == (self.input.output_size,)
    return self.regression_function(input_vector)

  def get_classification(self, input_vector):
    return numpy.argmax(self.get_regression(input_vector))

  def input_vector(self):
    return self.input.output

  def all_layers(self):
    return self.output.ancestors()

  def learn(self, *args):
    if self.learning_rate < numpy.inf:
      self.learning_rate *= 0.5 ** (1/self.learning_rate_half_time)

    return self.predictor.learn(*args)

  def symbolic_updates(self):
    symbolic_updates = []

    for layer in self.updates:
      for component in layer.components():
        component_grad = theano.tensor.grad(cost=self.cost, wrt=component)
        new_component = component - self.learning_rate * component_grad
        symbolic_updates.append((component, new_component))

    return symbolic_updates

class FullFeedbackVector(object):
  def __init__(self, dimension):
    self.label = T.iscalar("label")
    self.dimension = dimension

  def compile_learning_function(self, net):
    self.learning_function = theano.function(
      inputs=[net.input_vector(), self.label],
      outputs=net.cost,
      updates=net.symbolic_updates())

  def learn(self, input_vector, label):
    assert 0 <= label < self.dimension
    return self.learning_function(input_vector, label)

  def negative_log_loss(self, layer):
    return - layer.output[self.label]

class RewardFeedbackVector(object):
  def __init__(self, dimension):
    self.label = T.iscalar("action") # aka action
    self.reward = T.dscalar("reward")
    self.dimension = dimension

  def compile_learning_function(self, net):
    self.learning_function = theano.function(
      inputs=[net.input_vector(), self.label, self.reward],
      outputs=net.cost,
      updates=net.symbolic_updates())

  def learn(self, input_vector, label, reward):
    assert 0 <= label < self.dimension
    assert 0 <= reward <= 1
    return self.learning_function(input_vector, label, reward)

  def distance_loss(self, layer):
    return T.mean((layer.output[self.label] - self.reward)**2)