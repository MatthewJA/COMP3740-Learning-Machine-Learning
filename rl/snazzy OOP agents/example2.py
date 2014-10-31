from SnazzyNets import *
import numpy as np
import random

# class DenoisingAutoencoderAgent(SnazzyNet):
#   def __init__(self
#         , input_dimension
#         , hidden_dimension
#         , output_dimension
#         , learning_rate = 0.1
#         , corruption = 0.3):

#     self.learning_rate = learning_rate
#     self.corruption = corruption

#     self.input_vector = InputVector(input_dimension)
#     noise_layer = Noise(self.input_vector, corruption)
#     self.hidden_layer = LinearTransformation(noise_layer, hidden_dimension)
#     sigmoid_layer_1 = SigmoidLayer(self.hidden_layer)
#     reconstruction = LinearTransformation(sigmoid_layer_1, input_dimension)
#     self.reconstruction_layer = SigmoidLayer(reconstruction_layer)
#     self.label_layer = LinearTransformation(reconstruction, output_dimension)

#     reconstruction_cost = SnazzyNet.negative_log_loss(self.input_vector,
#                               self.reconstruction_layer)

#     error_cost = SnazzyNet.difference_thing(self.label_layer, self.)

#     self.reward_modulation = Parameter(1)

#     self.cost = (self.reconstruction_cost * self.reward_modulation
#           + 0.2 * self.error_cost * (1 - self.reconstruction)
#           + 0.4 * reconstruction.l1_regularization()
#           + 0.6 * self.label_layer.l1_regularization())


class LogisticRegressionNet(SnazzyNet):
  def __init__(self, input_dimension, output_dimension, learning_rate = 0.1):
    self.input = InputVector(input_dimension)
    self.transformation = LinearTransformation(self.input, output_dimension)
    self.sigmoid = SigmoidLayer(self.transformation)

    self.predictor = FullFeedbackVector(self, output_dimension)
    self.cost = self.predictor.negative_log_loss(self.sigmoid)
    self.output_object = self.sigmoid
    self.learning_rate = 0.1

    self.updates = [self.transformation]

    SnazzyNet.__init__(self)


class LogisticRegressionCBNet(LogisticRegressionNet):
  def __init__(self, input_dimension, output_dimension, learning_rate = 0.1):
    self.input = InputVector(input_dimension)
    self.transformation = LinearTransformation(self.input, output_dimension)
    self.sigmoid = SigmoidLayer(self.transformation)

    self.predictor = RewardFeedbackVector(output_dimension)
    self.cost = (self.predictor.distance_loss(self.transformation) +
                                  0.01 * self.transformation.l1_regularization())
    self.output_object = self.sigmoid
    self.learning_rate = 0.1

    self.updates = [self.transformation]

    SnazzyNet.__init__(self)



data = [(np.array([0,0]), 0), (np.array([0,1]), 1),
        (np.array([1,0]), 2)]

lr_cb = LogisticRegressionCBNet(2, 3)


for (input_vector, label) in data*100:
  if random.random() < 0.1:
    action = random.randrange(3)
  else:
    action = lr_cb.get_classification(input_vector)

  print action == label
  lr_cb.learn(input_vector, action, action == label)

correct = 0
for (input_vector, label) in data:
  correct += lr_cb.get_classification(input_vector) == label

print correct