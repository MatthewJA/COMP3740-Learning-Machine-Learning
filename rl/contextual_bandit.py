#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contextual bandit agent using k-nearest-neighbours.

The problem is easier when you have more easily distinguishable observations.

Principal Author: Buck Shlegeris
"""

from __future__ import division

import pylab
import random

class ContextualBanditAgent(object):
  # self.experiences is a list of (observation, action, reward) tuples.
  def __init__(self, num_actions, k = 10, epsilon = 0.1):
    self.num_actions = num_actions
    self.experiences = []
    self.epsilon = 0.1
    self.k = k

  def act(self, observation):
    if random.random() < self.epsilon or len(self.experiences) == 0:
      return random.randrange(self.num_actions)
    else:
      relevant_experiences = sorted([
        (self.distance(observation, x[0]), x) for x in self.experiences])[:self.k]
      expected_results = { x:[] for x in range(self.num_actions)}
      for thing in relevant_experiences:
        (observation, action, reward) = thing[1]
        expected_results[action].append(reward)
      values = {x : sum(expected_results[x])/len(expected_results[x])
                  for x in expected_results
                  if expected_results[x]}
      return random.choice([x for x in values if values[x] == max(values.values())])

  def curried_distance(self, x):
    def f(y,z):
      return self.distance(x, y[0]) - self.distance(x, z[0])
    return f

  def distance(self, x, y):
    return abs(x-y)

  def learn(self, observation, action, reward):
    self.experiences.append((observation, action, reward))

class ContextualBandit(object):
  def __init__(self, states):
    self.states = states

  def getTurn(self):
    state = random.randrange(len(self.states))
    observation = random.normalvariate(self.states[state], 1)
    def reward(action):
      return int(action == state)
    return (observation, reward)

def test_different_observation_distributions():
  for states in [[0,0.5], [0,1], [0,2], [0,3]]:
    environment = ContextualBandit(states)
    length_of_trial = 200
    num_trials = 200
    reward_group_size = 10
    rewards = [0] * (length_of_trial // reward_group_size)

    for trial in range(num_trials):
      agent = ContextualBanditAgent(len(states))
      print trial
      for run in range(length_of_trial):
        observation, rewardFunction = environment.getTurn()
        action = agent.act(observation)
        reward = rewardFunction(action)
        agent.learn(observation, action, reward)

        rewards[run // reward_group_size] += reward

    pylab.plot([(i*reward_group_size, x/num_trials)
                            for (i,x) in enumerate(rewards)], label=str(states))

  pylab.legend(loc='upper left')
  pylab.show()

def test_different_k():
  for k in [10,30,50]:
    environment = ContextualBandit([0,2,4])
    length_of_trial = 400
    num_trials = 300
    rewards = [0] * length_of_trial

    for trial in range(num_trials):
      print trial
      agent = ContextualBanditAgent(3, k)

      for run in range(length_of_trial):
        observation, rewardFunction = environment.getTurn()
        action = agent.act(observation)
        reward = rewardFunction(action)
        agent.learn(observation, action, reward)

        rewards[run] += reward

    pylab.plot([x/num_trials for x in rewards], label=str(k))

  pylab.legend(loc='upper left')
  pylab.show()

test_different_observation_distributions()