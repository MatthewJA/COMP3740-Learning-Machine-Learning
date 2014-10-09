from __future__ import division

import math

import numpy

import cart_pole
import mdp_da

if __name__ == '__main__':

  cart = cart_pole.Cart(math.pi/2 - 0.1)

  input_dimension = len(cart.get_state()) # The length of a state vector
  hidden_dimension = 200 # Arbitrary, at present
  output_dimension = 3 # 3 possible actions
  batch_size = 20

  agent = mdp_da.MDP_DA(input_dimension, hidden_dimension, output_dimension, gamma=0.2, learning_rate=0.3)

  lengths = []

  import cPickle
  import sys

  i = 0
  all_states = None
  all_actions = None
  all_rewards = None
  while True:
    state_info = cart_pole.get_states(agent, cart, 0.01, True)
    states, actions, rewards = map(numpy.asarray, zip(*state_info))
    if all_states is not None:
      all_states = numpy.concatenate((all_states, states), axis=0)
      all_actions = numpy.concatenate((all_actions, actions), axis=0)
      all_rewards = numpy.concatenate((all_rewards, rewards), axis=0)
    else:
      all_states = states
      all_actions = actions
      all_rewards = rewards

    lengths.append(len(state_info))
    print len(state_info)
    print >> sys.stderr, i, len(state_info), len(all_states), "|"
    i += 1

    for batch_index in xrange(0, len(all_states), batch_size):
      agent.train_model_once(all_states[batch_index:batch_index+batch_size], all_actions[batch_index:batch_index+batch_size], all_rewards[batch_index:batch_index+batch_size])

    if i%100 == 0:
      with open("cartDA.pickle", "w") as f:
        cPickle.dump(agent, f)

  # state = cart.get_state()
  # state = numpy.asarray([state])

  def get_action(cart):
    return cart_pole.get_action(agent, cart)

  cart.reset()
  cart_pole.animate_cart(cart, get_action)
