"""
A maze with states represented by numbers from the MNIST dataset.

Principal Author: Matthew Alger
"""

from __future__ import division

import random

import numpy

import lib.mnist as mnist

class Maze(object):
    """
    A maze with states represented by numbers from the MNIST dataset.

    Actions taken in the maze can be 0, 1, 2, and 3, which represent right, up,
    left, and down respectively.
    """

    def __init__(self, grid, start, goal):
        """
        grid: A 2D list of digits, each representing a maze state.
            e.g. [[1, 2, 3],
                  [4, 5, 6]]
        start: Which digit to start on.
        goal: Which digit to reach.
            e.g. If for the maze above, the start was 1 and the goal was 6,
                the optimal solutions would be 003, 300, and 030.
        """

        self.grid = grid
        self.start_digit = start
        self.goal_digit = goal

        self.start = None
        self.goal = None
        for y, i in enumerate(grid):
            if self.start_digit in i:
                self.start = (i.index(self.start_digit), y)
            if self.goal_digit in i:
                self.goal = (i.index(self.goal_digit), y)
        if self.start is None:
            raise ValueError("Starting digit not in maze.")
        if self.goal is None:
            raise ValueError("Goal digit not in maze.")

        self.position = self.start
        self.height = len(self.grid)
        self.width = 0 if self.height == 0 else len(self.grid[0])

        self.load_mnist_data()

    def move(self, direction):
        """
        Move around in the maze.

        direction: 0, 1, 2, or 3, representing right, up, left, and down
            respectively.
        """

        x, y = self.position
        if direction == 0 and self.position[0] < self.width-1:
            self.position = x+1, y
        elif direction == 1 and self.position[1] > 0:
            self.position = x, y-1
        elif direction == 2 and self.position[0] > 0:
            self.position = x-1, y
        elif direction == 3 and self.position[1] < self.height-1:
            self.position = x, y+1
        elif direction not in {0, 1, 2, 3}:
            raise ValueError("Direction must be an integer from 0 to 3 "
                "inclusive. Received: {}".format(direction))

    def load_mnist_data(self):
        """
        Load the MNIST dataset into self.mnist, a dictionary of MNIST data
        with labels as keys.
        """

        labels = mnist.load_training_labels()
        images = mnist.load_training_images(div=256)

        self.mnist = {i:[] for i in xrange(10)}
        for index, label in enumerate(labels):
            self.mnist[int(label)].append(images[index])

    def get_state(self):
        """
        Returns an image from the MNIST dataset representing where we are.
        """

        x, y = self.position
        digit = self.grid[y][x]

        return random.choice(self.mnist[digit])

    def reset(self):
        """
        Reset the maze.
        """

        self.position = self.start

    def get_reward(self):
        """
        Get the reward for being where we are.
        """

        if self.finished():
            return 1
        return 0

    def finished(self):
        """
        Return whether or not we have finished navigating the maze.
        """

        return self.position == self.goal

def get_action(agent, maze):
    """
    Get an action to take based on the state of the maze.

    agent: An object with a get_expected_rewards method, that takes a state
        vector (2d NumPy array with one row) and returns a NumPy array of
        expected rewards.
    """

    state = numpy.asarray([maze.get_state()])
    expected_rewards = agent.get_expected_rewards(state)
    action = numpy.argmax(expected_rewards)
    return action

def get_states(agent, maze, epsilon=0.1):
    """
    Move through the maze according to the agent, and return tuples of the form
    [state, action, discounted_future_reward].

    agent: An object with a get_expected_rewards method as described in
        get_action, and a gamma property describing how fast discounted future
        reward falls off.
    maze: A Maze object.
    epsilon: Percentage chance of taking a random action instead of performing
        the action given by the agent. Optional (default 0.1).
    """

    maze.reset()

    lists = []
    while not maze.finished():
        state = maze.get_state()
        action = get_action(agent, maze)
        if random.random() < epsilon:
            action = random.randrange(4)
        maze.move(action)
        reward = 0 # Temporarily - we'll update this later to be discounted
                   # future reward.
        lists.append([state, action, reward])
    if lists:
        lists[-1][2] = 1 # Solving the maze gives us reward of 1.

    tuples = []
    last_reward = 0
    while lists:
        state, action, reward = lists.pop()
        discounted_future_reward = (last_reward * agent.gamma + reward)
        last_reward = discounted_future_reward
        tuples.append((state, action, discounted_future_reward))

    return tuples

# def get_states_q_learning(agent, maze, epsilon=0.1):
#     maze.reset()

#     lists = []
#     while not maze.finished():
#         state = maze.get_state()
#         action = get_action(agent, maze)
#         if random.random() < epsilon:
#             action = random.randrange(4)
#         maze.move(action)
#         reward = 0 # Temporarily - we'll update this later to be estimated
#                    # future reward
#         lists.append([state, action, reward])
#     if lists:
#         lists[-1][2] = 1 # Solving the maze gives us reward of 1.

#     tuples = []
#     last_reward = 0
#     while lists:
#         state, action, reward = lists.pop()
#         discounted_future_reward = (last_reward * agent.gamma + reward)
#         last_reward = discounted_future_reward

#         new_value = agent.expected_reward(state)
#         max(agent.expected_)

#         tuples.append((state, action, discounted_future_reward))

#     return tuples
