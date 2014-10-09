"""
A maze with states represented by numbers from the MNIST dataset.
"""

from __future__ import division

import random

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
        for x, i in enumerate(grid):
            if self.start_digit in i:
                self.start = (x, i.index(self.start_digit))
            if self.goal_digit in i:
                self.goal = (x, i.index(self.goal_digit))
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
        else:
            raise ValueError("Direction must be an integer from 0 to 1.")

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

        

if __name__ == '__main__':
    maze = Maze([[1, 2, 3], [4, 5, 6]], 1, 6)
    maze.load_mnist_data()