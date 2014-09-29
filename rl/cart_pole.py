#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cart-pole problem.

Principal Author: Matthew Alger
"""

from __future__ import division

import random
from math import cos, acos, pi, sin

class Cart(object):
	"""
	A cart which moves around and can either accelerate left, accelerate right,
	or not accelerate at any given step.
	"""

	def __init__(self, x, p):
		"""
		x: x position of cart
		p: angle of pole in radians (0 to π)
		"""
		self.x = x # position in units
		self.p = p # pole angle in radians
		self.vx = 0 # velocity in units/step
		self.a = 0 # acceleration in units/step^2
		self.l = 100 # length of pole in units
		self.m = 1 # pole mass in kg
		self.top_speed = self.l/5 # maximum velocity in units/step
		self.vp = 0 # angular pole velocity in radians/step
		self.g = 0.1 # gravitational acceleration in units/step^2

	def step(self, action):
		"""
		Simulate the cart.

		Action is an acceleration. It can be 1, -1, or 0.
		"""

		# We need this to estimate inertia.
		x_pole_position = self.l * cos(self.p)

		# The cart moves...
		self.a = action
		self.vx = min(self.vx + self.a, self.top_speed)
		self.x += self.vx

		# The pole shifts position...
		# The shift will be related to the cosine of the angle.
		# A perfectly vertical pole will maintain its x position.
		# A perfectly horizontal pole will move completely to a new x position.
		# Mass should dampen the movement because of inertia.
		new_x = x_pole_position + min(self.vx, self.vx * cos(self.p/2) / self.m)
		self.p = acos(max(-self.l, min(self.l, (new_x - self.x)))/self.l)

		# The pole falls.
		# Calculate torque on the pole.
		gravitational_force = self.m * self.g
		perpendicular_force = -gravitational_force * cos(self.p)
		torque = perpendicular_force * self.l/2 # Average centre of mass = l/2
		# Angular acceleration is then given by Newton's 2nd Law.
		moment_of_inertia = self.m * self.l**2/3 # Moment of inertia of pole
		angular_acceleration = torque/moment_of_inertia
		self.vp += angular_acceleration
		self.p = max(0, min(pi, self.p + self.vp))

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

def update(cart):
	cart.step(0)#(random.randrange(-1, 2))

def animate_cart(cart):
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

		update(cart)
		draw(pygame, screen, cart)

		pygame.display.flip()
		clock.tick(fps)

if __name__ == '__main__':
	cart = Cart(0, pi/2+0.1)
	animate_cart(cart)