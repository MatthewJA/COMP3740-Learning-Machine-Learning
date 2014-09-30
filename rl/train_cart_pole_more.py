from denoising_autoencoder_cart_pole import Cart_Pole_DA
import cart_pole
import cPickle
import sys
import numpy

with open("cartDA.pickle") as f:
  agent = cPickle.load(f)
  cart = cart_pole.Cart(1)

lengths = []


i = 0
while True:
  state_info = cart_pole.get_states(agent, cart, 0)
  states, actions, rewards = map(numpy.asarray, zip(*state_info))
  lengths.append(len(state_info))
  print len(state_info)
  print >> sys.stderr, i, len(state_info)
  i += 1


  agent.train_model_once(states, actions, rewards)