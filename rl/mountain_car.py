#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import random
import pylab

# We're on a quadratic slope: y = c x^2
# Gradient(x) = 2x, so acceleration = -2cgx

# action \in {-1, 0, 1}


def get_output_function(variables):
  def closeness(vector1, vector2):
    return 0.9 ** ((x-y)**2 for (x,y) in zip(vector1, vector2))

  input_vector = list([x] for x in variables[0][1])
  for variable_name, variable_range in variables[1:]:
    input_vector = [x + [y] for x in input_vector for y in variable_range]

  def output_function(self):
    actual_position = [self.__dict__[name] for (name, _) in variables]
    return [closeness(actual_position, other_position)
              for other_position in input_vector]

  return output_function

class MountainCar(object):
  """
  A cart which moves around and can either accelerate left, accelerate right,
  or not accelerate at any given step.
  """

  def __init__(self):
    self.reset()

  def reset(self):
    """
    Reset the dynamic properties of the cart.
    """

    self.c = 0.01
    self.g = 9.8
    self.accelerator = 0.5
    self.boundary = 100
    self.position = 20
    self.velocity = 0

  def step(self, action):
    """
    Simulate the cart.

    action gives the acceleration. It can be 0, 1, or 2.
    1 is subtracted from this to get the acceleration.
    """

    action -= 1

    self.velocity += action * self.accelerator - 2 * self.g * self.position * self.c
    self.position += self.velocity


  def get_state(self):
    """
    Return a list encoding the state of the cart.
    """

    # We have five RBFs per dimension.
    # Along the x axis, we will have RBFs at -400, -200, 0, 200, 400.
    # Along the vx axis, we will have RBFs at -top_speed, -top_speed/2,
    # 0, top_speed/2, top_speed.
    # Along the p axis, we will have RBFs at 0, π/4, π/2, 3π/4, π.
    # Along the vp axis, we will have RBFs at -top_vp, -top_vp/2,
    # 0, top_vp/2, top_vp.
    # We will have 5^4 in total.
    rbf_values = []
    for x in (-100, -50, 0, 50, 100):
      for vx in (-40, -20, 0, 20, 40):
        sq_dist = (
          (self.x - x)**2 +
          (self.vx - vx)**2 +
          (self.p - p)**2 +
          (self.vp - vp)**2)

        rbf_value = exp(-sq_dist)
        rbf_values.append(rbf_value)
    return rbf_values

  def game_over(self):
    """
    Whether the pole has hit the cart.
    """

    return abs(self.boundary) > self.boundary

def update(cart, action):
  cart.step(action)

def get_action(agent, cart):
  """
  Get an action to take based on the state of the cart.
  """

  state = cart.get_state()
  state = numpy.asarray([state])
  expected_rewards = agent.get_expected_rewards(state)
  action = numpy.argmax(expected_rewards)
  return action

def get_states(agent, cart, epsilon=0.1):
  """
  Run the cart according to the agent, and return tuples of the form
  [state, action, discounted_future_reward].
  """

  cart.reset()

  lists = []
  while not cart.game_over():
    state = cart.get_state()
    action = get_action(agent, cart)
    if random.random() < epsilon:
      action = random.randrange(3)
    cart.step(action)
    reward = 0 # for now
    lists.append([state, action, reward])
  if lists:
    lists[-1][2] = 1 # Getting out gives us +1 reward.

  tuples = []
  last_reward = 0
  while lists:
    state, action, reward = lists.pop()
    discounted_future_reward = (last_reward * agent.gamma + reward)
    last_reward = discounted_future_reward
    tuples.append((state, action, discounted_future_reward+1))

  return tuples
