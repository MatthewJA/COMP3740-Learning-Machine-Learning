from SnazzyNets import *
import numpy as np
import random

class MLPFullFeedbackNet(SnazzyNet):
  def __init__(self, input_dimension, hidden_layer_size, output_dimension,
                        learning_rate = 0.5, learning_rate_half_time=np.inf):

    self.input = InputVector(input_dimension)
    self.hidden_layer = NeuralLayer(self.input, hidden_layer_size)
    self.output_layer = NeuralLayer(self.hidden_layer, output_dimension)

    self.predictor = FullFeedbackVector(output_dimension)

    self.cost = (self.predictor.negative_log_loss(self.output_layer) +
                0.001 * self.hidden_layer.l1_regularization() +
                0.001 * self.output_layer.l1_regularization())

    self.learning_rate = learning_rate
    self.learning_rate_half_time = learning_rate_half_time

    self.updates = [self.hidden_layer, self.output_layer]


    SnazzyNet.__init__(self)

def xor_test():
  print "starting xor test, we expect to get 3/4 correct: ",
  mlp = MLPFullFeedbackNet(2, 5, 2)

  xor_data = [(np.array([0,0]), 0), (np.array([0,1]), 1),
          (np.array([1,0]), 1), (np.array([1,1]), 1)]

  for (input_vector, label) in xor_data*2000:
    print mlp.learn(input_vector, label)

  correct = 0
  for (input_vector, label) in xor_data:
    correct += mlp.get_classification(input_vector) == label

  print correct

xor_test()