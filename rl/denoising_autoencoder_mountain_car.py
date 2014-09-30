from __future__ import division

import math

import numpy
import theano
import pylab

from denoising_autoencoder import Denoising_Autoencoder, test_DA
import cart_pole
import mountain_car
import denoising_autoencoder_cart_pole

if __name__ == '__main__':

  cart = cart_pole.Cart(math.pi/2+0.1)

  input_dimension = len(cart.get_state()) # The length of a state vector
  hidden_dimension = 429 # Arbitrary, at present
  output_dimension = 3 # 3 possible actions

  agent = Cart_Pole_DA(input_dimension, hidden_dimension, output_dimension, gamma=0.9)

  lengths = []

  import cPickle
  import sys

  i = 0
  while True:
    state_info = cart_pole.get_states(agent, cart)
    states, actions, rewards = map(numpy.asarray, zip(*state_info))
    lengths.append(len(state_info))
    print len(state_info)
    print >> sys.stderr, i, len(state_info)
    i += 1


    agent.train_model_once(states, actions, rewards)

    if i%1000 == 0:
      with open("cartDA.pickle", "w") as f:
        cPickle.dump(agent, f)