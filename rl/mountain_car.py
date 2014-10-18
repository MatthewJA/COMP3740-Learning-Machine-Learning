#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import random
import pylab
from math import cos, acos, pi, sin, exp, sqrt

# We're on a quadratic slope: y = c x^2
# Gradient(x) = 2x, so acceleration = -2cgx

# action \in {-1, 0, 1}

class MountainCar(object):
  """
  A car which moves around and can either accelerate left, accelerate right,
  or not accelerate at any given step.
  """

  def __init__(self):
    self.reset()

  def reset(self):
    """
    Reset the dynamic properties of the car.
    """

    self.c = 0.01
    self.g = 9.8
    self.accelerator = 0.5
    self.boundary = 10
    self.position = 0
    self.velocity = 0

  def step(self, action):
    """
    Simulate the car.

    action gives the acceleration. It can be 0, 1, or 2.
    1 is subtracted from this to get the acceleration.
    """

    action -= 1

    self.velocity += action * self.accelerator - 2 * self.g * self.position * self.c
    self.position += self.velocity


  def get_state(self):
    """
    Return a list encoding the state of the car.
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
    for position in (-100, -50, 0, 50, 100):
      for velocity in (-40, -20, 0, 20, 40):
        sq_dist = (
          (self.position - position)**2 +
          (self.velocity - velocity)**2)

        rbf_value = exp(-sq_dist)
        rbf_values.append(rbf_value)

    return rbf_values

  def game_over(self):
    """
    Whether the pole has hit the car.
    """

    return abs(self.position) > self.boundary

def update(car, action):
  car.step(action)

def get_action(agent, car):
  """
  Get an action to take based on the state of the car.
  """

  state = car.get_state()
  state = numpy.asarray([state])
  expected_rewards = agent.get_expected_rewards(state)
  action = numpy.argmax(expected_rewards)
  return action

def get_states(agent, car, epsilon=0.1):
  """
  Run the car according to the agent, and return tuples of the form
  [state, action, discounted_future_reward].
  """

  car.reset()

  lists = []
  while not car.game_over():
    state = car.get_state()
    action = get_action(agent, car)
    if random.random() < epsilon:
      action = random.randrange(3)
    car.step(action)
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
    tuples.append((state, action, discounted_future_reward))


  return tuples
