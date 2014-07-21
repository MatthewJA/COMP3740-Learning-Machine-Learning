"""
Classify data using logistic regression.
"""

import theano

import mnist

class Classifier(object):
	"""
	Learns to classify and then classifies data.

	The classifier will hold a weight matrix W and a bias vector b that it
	uses to classify data. The matrix and the vector comprise the model, and
	can be thought of as the ``knowledge'' of the classfier. Through gradient
	descent the values of W and b will be found and tuned to minimise
	classification error (loss).
	"""

	def __init__(self, input_dimension, output_dimension):
		"""
		The input dimension is a number representing the dimensions of the input
		vectors. For example, a 28×28 image would be represented by a
		784-dimensional vector, so this parameter would be 784.

		The output dimension is the number of labels that the classifier will
		classify. For example, if you want to detect digits, you would have
		labels {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, which is a 10-dimensional vector,
		so this parameter would be 10.
		"""

		self.input_dimension = input_dimension
		self.output_dimension = output_dimension

		self.initialise_weight_matrix()

	def initialise_weight_matrix(self):
		matrix = theano.numpy.zeros(
				()
			)