from mountain_car import *
import mdp_da
import numpy
import cPickle
import sys


if __name__ == '__main__':

  mountain_car = MountainCar()

  input_dimension = len(mountain_car.get_state()) # The length of a state vector
  hidden_dimension = 429 # Arbitrary, at present
  output_dimension = 3 # 3 possible actions

  agent = mdp_da.MDP_DA(input_dimension, hidden_dimension, output_dimension, gamma=0.9)

  lengths = []

  i = 0
  while True:
    state_info = get_states(agent, mountain_car)
    states, actions, rewards = map(numpy.asarray, zip(*state_info))
    lengths.append(len(state_info))
    print len(state_info)
    print >> sys.stderr, i, len(state_info)
    i += 1

    agent.train_model_once(states, actions, rewards)

    if i%1000 == 0:
      with open("mountain_carDA.pickle", "w") as f:
        cPickle.dump(agent, f)
