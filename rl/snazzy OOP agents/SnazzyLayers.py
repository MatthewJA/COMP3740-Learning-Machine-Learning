
import numpy
import theano
import theano.tensor as T

class Layer(object):
  def components(self):
    return []

  def ancestors(self):
    return [x for x in y.ancestors() for y in self.output]

  # TODO
  def negative_log_loss(self, predictor):
    return - T.log(self.output[predictor.label])

  def get_component_values(self):
    self.get_component_values = theano.function(inputs=[], outputs=self.components())
    return self.get_component_values()

class LinearTransformation(Layer):
  def __init__(self, input_object, output_size):
    # assert input_object.output_dimension =
    self.input_object = input_object
    self.output_size = output_size
    self.W = theano.shared(
            value=numpy.zeros((input_object.output_size, output_size),
            dtype=theano.config.floatX),
            name='W',
            borrow=True)
    self.b = theano.shared(
            value=numpy.zeros((output_size,),
            dtype=theano.config.floatX),
            name='b',
            borrow=True)

    self.output = T.dot(self.input_object.output, self.W) + self.b

    self.l1_regularization_values = theano.function(
                  inputs=[], outputs=self.l1_regularization())

  def components(self):
    return [self.W, self.b]

  def l1_regularization(self):
    return T.mean(abs(self.W)) + T.mean(abs(self.b))

class NeuralLayer(LinearTransformation):
  def __init__(self, input_object, output_size):
    LinearTransformation.__init__(self, input_object, output_size)

    self.linear_transformation_output = self.output
    self.sigmoid = SigmoidLayer(self)
    self.output = self.sigmoid.output
    self.output_size = self.sigmoid.output_size

class SigmoidLayer(Layer):
  def __init__(self, input_object):
    self.input_object = input_object
    self.output_size = input_object.output_size

    # TODO: comment this
    self.output = T.nnet.softmax(self.input_object.output)[0]

class InputVector(object):
  def __init__(self, dimension):
    self.output_size = dimension
    self.output = theano.tensor.dvector("input")

  # TODO
  def reconstruction_cost(self, symbolic_reconstruction):
    pass