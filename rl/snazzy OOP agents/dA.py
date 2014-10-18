import SnazzyNets

class DenoisingAutoencoder(SnazzyNet):
  def __init__(self
              , input_dimension
              , hidden_dimension
              , output_dimension
              , learning_rate = 0.1
              , corruption = 0.3):

    self.input_vector = InputVector(input_dimension)
    noise_layer = Noise(self.input_vector, corruption)
    self.hidden_layer = LinearTransformation(noise_layer, hidden_dimension)
    sigmoid_layer_1 = SigmoidLayer(self.hidden_layer)
    reconstruction = LinearTransformation(sigmoid_layer_1, input_dimension)
    self.reconstruction_layer = SigmoidLayer(reconstruction_layer)
    self.label_layer = LinearTransformation(self.)