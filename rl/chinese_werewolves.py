from __future__ import division
import random

# Chinese Whispers

"""
Principal Author: Buck Shlegeris
"""

class MultiBanditAgent(object):
  def __init__(self, n_observations, n_outputs, learning_rate = 0.8, epsilon = 0.1):
    self.n_outputs = n_outputs
    self.n_observations = n_observations
    self.values = [[0] * n_outputs for x in range(n_observations)]
    self.learning_rate = learning_rate
    self.epsilon = epsilon

  def make_decision(self, observation):
    if random.random() > self.epsilon:
      bests = [idx for (idx, x) in enumerate(self.values[observation])
                                           if x == max(self.values[observation])]
      return random.choice(bests)
    else:
      return random.choice(range(self.n_outputs))

  def update(self, observation, action, reward):
    self.values[observation][action] = (
        self.values[observation][action] * (1 - self.learning_rate)
      + reward * self.learning_rate)


seer = MultiBanditAgent(2,2)
watchers = [MultiBanditAgent(2,2) for x in range(1)]

for a in range(100):
  observation = int(random.random() > 0.5)
  seer_action = seer.make_decision(observation)

  watcher_actions = []
  previous_action = seer_action

  for watcher in watchers:
    previous_action = watcher.make_decision(previous_action)
    watcher_actions.append(previous_action)

  reward = int(watcher_actions[-1] == observation)

  seer.update(observation, seer_action, reward)
  for (watcher, act, obs) in zip(watchers, watcher_actions, [seer_action]+watcher_actions):
    watcher.update(obs, act, reward)

  print "observation: %d, reward %d"%(
              observation, reward)

