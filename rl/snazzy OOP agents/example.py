from SnazzyNets import *
import numpy as np
import random

class LogisticRegressionCBNet(SnazzyNet):
  def __init__(self, input_dimension, output_dimension, learning_rate = 0.5,
                            learning_rate_half_time=np.inf):
    self.input = InputVector(input_dimension)
    self.hidden_layer = HiddenLayer(self.input, output_dimension)
    self.sigmoid = SigmoidLayer(self.hidden_layer)

    self.predictor = RewardFeedbackVector(output_dimension)

    # comment on this
    self.cost = (self.predictor.distance_loss(self.sigmoid) +
                0.0000001 * self.hidden_layer.l1_regularization())
    self.output_object = self.sigmoid
    self.learning_rate = learning_rate
    self.learning_rate_half_time = learning_rate_half_time

    self.updates = [self.hidden_layer]

    SnazzyNet.__init__(self)

