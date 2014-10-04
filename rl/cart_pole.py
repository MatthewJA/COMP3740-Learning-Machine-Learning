#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cart-pole problem.

Principal Author: Matthew Alger
"""

from __future__ import division

import random
from math import cos, acos, pi, sin, exp, sqrt

import numpy

class Cart(object):
	"""
	A cart which moves around and can either accelerate left, accelerate right,
	or not accelerate at any given step.
	"""

	def __init__(self, p=pi/2):
		self.x = 0 # position in units
		self.p = p # pole angle in radians
		self.init_p = p # position to reset to
		self.vx = 0 # velocity in units/step
		self.a = 0 # acceleration in units/step^2
		self.l = 100 # length of pole in units
		self.m = 1 # pole mass in kg
		self.top_speed = self.l/5 # maximum velocity in units/step
		self.vp = 0 # angular pole velocity in radians/step
		self.top_vp = pi/2 # maximum angular pole velocity in radians/step
		self.g = 5 # gravitational acceleration in units/step^2
		self.maxx = 500
		self.minx = -500

	def reset(self, randomp=False):
		"""
		Reset the dynamic properties of the cart.
		"""

		self.x = 0
		self.p = self.init_p if not randomp else (
			random.random()*pi)
		self.vx = 0
		self.a = 0
		self.vp = 0

	def step(self, action):
		"""
		Simulate the cart.

		action gives the acceleration. It can be 0, 1, or 2.
		1 is subtracted from this to get the acceleration.
		"""

		action -= 1

		# The cart moves...
		self.a = action
		self.vx = min(self.vx + self.a, self.top_speed)
		old_x = self.x
		self.x += self.vx
		if self.x <= self.minx:
			self.x = self.minx
			self.vx = -self.vx
		elif self.x >= self.maxx:
			self.x = self.maxx
			self.vx = -self.vx

		# ...And the pole moves with it.
		back_force = -sin(self.p) * self.m * self.a # maybe this should be v not a?
		forward_force = -cos(self.p) * self.m * self.g
		moment_of_inertia = self.m * (self.l/2)**2 / 3
		torque = forward_force - back_force
		ap = torque/moment_of_inertia
		self.vp += ap
		self.p += self.vp

		if self.p <= 0:
			self.p = 0
			self.vp = 0
		elif self.p >= pi:
			self.p = pi
			self.vp = 0

	def get_state(self):
		"""
		Return a list encoding the state of the cart.
		"""

		# We have five RBFs per dimension.
		# Along the x axis, we will have RBFs at -400, -200, 0, 200, 400.
		# Along the vx axis, we will have RBFs at -top_speed, -top_speed/2,
		#	0, top_speed/2, top_speed.
		# Along the p axis, we will have RBFs at 0, π/4, π/2, 3π/4, π.
		# Along the vp axis, we will have RBFs at -top_vp, -top_vp/2,
		#	0, top_vp/2, top_vp.
		# We will have 5^4 in total.
		rbf_values = []
		for x in (-400, -200, 0, 200, 400):
			for vx in (-self.top_speed, -self.top_speed/2, 0,
				self.top_speed/2, self.top_speed):
				for p in (0, pi/4, pi/2, 3*pi/4, pi):
					for vp in (-self.top_vp, -self.top_vp/2, 0,
						self.top_vp/2, self.top_vp):

						# We have an RBF here.
						# How far away are we?
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

		return self.p < 1e-15 or self.p > pi-1e-15

def draw(pygame, screen, cart):
	screen.fill(0x000000)
	pygame.draw.rect(screen, 0xFFFFFF, (
		cart.x - 100 + screen.get_width()/2,
		screen.get_height() - 200,
		200,
		50))
	pygame.draw.line(screen, 0xFFFFFF, (
		cart.x + screen.get_width()/2,
		screen.get_height() - 200), (
		cart.x + screen.get_width()/2 + cart.l * cos(cart.p),
		screen.get_height() - 200 - cart.l * sin(cart.p)), 3)

def update(cart, action):
	cart.step(action)

def animate_cart(cart, get_action):
	import sys

	import pygame
	import pygame.locals

	pygame.init()
	size = width, height = 1000, 500
	screen = pygame.display.set_mode(size)
	fps = 20
	clock = pygame.time.Clock()

	while True:
		for event in pygame.event.get():
			if event.type == pygame.locals.QUIT:
				pygame.quit()
				sys.exit()

		update(cart, get_action(cart))
		draw(pygame, screen, cart)

		pygame.display.flip()
		clock.tick(fps)

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
		lists[-1][2] = -1 # Falling over gives us -1 reward.

	tuples = []
	last_reward = 0
	while lists:
		state, action, reward = lists.pop()
		discounted_future_reward = (last_reward * agent.gamma + reward)
		last_reward = discounted_future_reward
		tuples.append((state, action, discounted_future_reward+1))

	return tuples

if __name__ == '__main__':
	cart = Cart(pi/2)
	def get_action(cart): return random.randrange(0, 3)
	animate_cart(cart, get_action)