import cart_pole
import mdp_da
import numpy

if __name__ == '__main__':

  cart = cart_pole.Cart(0.1)

  input_dimension = len(cart.get_state()) # The length of a state vector
  hidden_dimension = 429 # Arbitrary, at present
  output_dimension = 3 # 3 possible actions

  agent = mdp_da.MDP_DA(input_dimension, hidden_dimension, output_dimension, gamma=0.9)

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

  # state = cart.get_state()
  # state = numpy.asarray([state])
  # print agent.get_expected_rewards(state)

  def get_action(cart):
    return cart_pole.get_action(agent, cart)

  cart.reset()
  cart_pole.animate_cart(cart, get_action)
