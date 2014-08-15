from logistic_regression import *
import mnist
import pylab

if __name__ == "__main__":
  print "loading training images"
  images = mnist.load_training_images(format="theano")
  print "loading training labels"
  labels = mnist.load_training_labels(format="theano")

  print "loading test images"
  test_images = mnist.load_test_images(format="theano")
  print "loading test labels"
  test_labels = mnist.load_test_labels(format="theano")

  print "instantiating classifier"
  classifier = Classifier(images, labels, 28*28, 10)

  print "training...",
  self_accuracy, test_accuracy = classifier.train_model(20, 600, True, (test_images, test_labels))
  print " done."

  pylab.plot(self_accuracy, label="%% errors on training data")
  pylab.plot(test_accuracy, label="%% errors on test data")
  pylab.legend()
  pylab.show()

  pylab.plot(self_accuracy, label="%% errors on training data")
  pylab.plot(test_accuracy, label="%% errors on test data")

  print "Wrong {:.02%} of the time".format(classifier.calculate_wrongness(
                                test_images, test_labels))
  print "(On the training set, wrong {:.02%} of the time)".format(
      classifier.calculate_wrongness(images, labels))