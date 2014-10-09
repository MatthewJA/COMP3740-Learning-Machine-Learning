"""
Solve a maze with states given by images in the MNIST dataset using a
denoising autoencoder.

Principal Author: Matthew Alger
"""

from __future__ import division

import random

import numpy
import PIL

import lib.dlt_utils as utils
import mnist_maze
import mdp_da

def train_agent(agent, maze, epochs=100, epsilon=0.1):
    """
    Train an agent on a given maze.

    agent: The MDP_DA to train.
    maze: The maze to train the agent on.
    epochs: The number of times to run the maze.
    epsilon: The chance of taking a random action while training.
    """

    all_states = numpy.empty((0, agent.input_dimension))
    all_actions = numpy.empty((0,))
    all_rewards = numpy.empty((0,))
    for epoch in xrange(epochs):
        state_info = mnist_maze.get_states(agent, maze, epsilon)
        states, actions, rewards = map(numpy.asarray, zip(*state_info))
        all_states = numpy.concatenate((all_states, states), axis=0)
        all_actions = numpy.concatenate((all_actions, actions), axis=0)
        all_rewards = numpy.concatenate((all_rewards, rewards), axis=0)

        for batch_index in xrange(0, len(all_states), batch_size):
            agent.train_model_once(
                all_states[batch_index:batch_index+batch_size],
                all_actions[batch_index:batch_index+batch_size],
                all_rewards[batch_index:batch_index+batch_size])

        print "Epoch {}/{}: {} moves".format(
            epoch+1, epochs,
            len(state_info))

        if epoch % 10 == 0:
            image = PIL.Image.fromarray(utils.tile_raster_images(
                X=agent.weights.get_value(borrow=True).T,
                img_shape=(28, 28), tile_shape=(10, 30),
                tile_spacing=(1, 1)))
            image.save('../plots/mnist_maze.png')

if __name__ == '__main__':
    maze = mnist_maze.Maze(
        [[1, 2, 3],
         [3, 2, 4],
         [5, 2, 6],
         [7, 2, 8],
         [0, 2, 9]],
        1, 9)

    input_dimension = len(maze.get_state())
    hidden_dimension = 300 # A hyperparameter with unknown optimal value.
                           # For now, we'll just set it to an arbitrary value.
    output_dimension = 4 # 4 possible actions.
    batch_size = 10 # Hyperparameter - size of data batch to average across.
    gamma = 0.2 # Hyperparameter - discounted future reward decay.
    learning_rate = 0.01

    agent = mdp_da.MDP_DA(
        input_dimension,
        hidden_dimension,
        output_dimension,
        gamma=gamma,
        learning_rate=learning_rate)

    train_agent(agent, maze)