from SnazzyNets import *
import numpy as np
import random

class LogisticRegressionCBNet(SnazzyNet):
  def __init__(self, input_dimension, output_dimension, learning_rate = 0.5,
                            learning_rate_half_time=np.inf):
    self.input = InputVector(input_dimension)
    self.output_layer = NeuralLayer(self.input, output_dimension)

    self.predictor = RewardFeedbackVector(output_dimension)

    self.cost = (self.predictor.distance_loss(self.output_layer) +
                0.001 * self.output_layer.l1_regularization())

    self.learning_rate = learning_rate
    self.learning_rate_half_time = learning_rate_half_time

    self.updates = [self.output_layer]

    SnazzyNet.__init__(self)

class LogisticRegressionFullFeedbackNet(LogisticRegressionCBNet):
  def __init__(self, input_dimension, output_dimension, learning_rate = 0.5,
                            learning_rate_half_time=np.inf):
    LogisticRegressionCBNet.__init__(self, input_dimension, output_dimension,
                              learning_rate, learning_rate_half_time)

    self.predictor = FullFeedbackVector(output_dimension)

    self.cost = (self.predictor.negative_log_loss(self.output_layer) +
                      0.001 * self.output_layer.l1_regularization())

    self.compile_functions()
