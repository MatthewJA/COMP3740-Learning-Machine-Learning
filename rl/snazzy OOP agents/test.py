from SnazzyNets import *
import numpy as np
import random
from logistic_regression import *

def simple_test():
  data = [(np.array([0,0]), 0), (np.array([0,1]), 1),
        (np.array([1,0]), 2)]

  lr_cb = LogisticRegressionCBNet(2, 3, 1)

  print "starting simple test, we expect to get 3/3 correct:",
  for (input_vector, label) in data*100:
    if random.random() < 0.4:
      action = random.randrange(3)
    else:
      action = lr_cb.get_classification(input_vector)

    lr_cb.learn(input_vector, action, action == label)

  correct = 0
  for (input_vector, label) in data:
    correct += lr_cb.get_classification(input_vector) == label

  print correct

def or_test():
  print "starting or test, we expect to get 4/4 correct:",
  lr_cb = LogisticRegressionCBNet(2, 2, 1, 20)

  or_data = [(np.array([0,0]), 0), (np.array([0,1]), 1),
          (np.array([1,0]), 1), (np.array([1,1]), 1)]

  for (input_vector, label) in or_data*20:
    if random.random() < 0.3:
      action = random.randrange(2)
    else:
      action = lr_cb.get_classification(input_vector)

    lr_cb.learn(input_vector, action, int(action == label))

  correct = 0
  for (input_vector, label) in or_data:
    correct += lr_cb.get_classification(input_vector) == label

  print correct

def xor_test():
  print "starting xor test, we expect to get 3/4 correct: ",
  lr_cb = LogisticRegressionCBNet(2, 2, 1, 20)

  xor_data = [(np.array([0,0]), 0), (np.array([0,1]), 1),
          (np.array([1,0]), 1), (np.array([1,1]), 0)]

  for (input_vector, label) in xor_data*2000:
    if random.random() < 0.3:
      action = random.randrange(2)
    else:
      action = lr_cb.get_classification(input_vector)

    lr_cb.learn(input_vector, action, int(action == label))

  correct = 0
  for (input_vector, label) in xor_data:
    correct += lr_cb.get_classification(input_vector) == label

  print correct

def full_feedback_or_test():
  print "starting or test with full feedback, we expect to get 4/4 correct:",
  lr_ff = LogisticRegressionFullFeedbackNet(2, 2, 1, 20)

  or_data = [(np.array([0,0]), 0), (np.array([0,1]), 1),
        (np.array([1,0]), 1), (np.array([1,1]), 1)]

  for (input_vector, label) in or_data*20:
    lr_ff.learn(input_vector, label)

  correct = 0
  for (input_vector, label) in or_data:
    correct += lr_ff.get_classification(input_vector) == label

  print correct

simple_test()
or_test()
xor_test()
full_feedback_or_test()