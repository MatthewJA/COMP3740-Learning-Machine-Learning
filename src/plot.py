#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot a value over time as it is calculated.
"""

from __future__ import division

from random import randrange

import pygame
import pygame.locals

def plot_over_iterators(iterators, labels=()):
	"""
	Plots values as yielded by iterators.

	iterator: A list of iterators which yield percentages.
	"""

	width = height = 500

	pygame.init()
	screen = pygame.display.set_mode((width, height))
	font = pygame.font.SysFont("sans-serif", 30)

	points = [[] for i in xrange(len(iterators))]
	colours = [(randrange(256), randrange(256), randrange(256))
		for i in xrange(len(iterators))]
	labels = [font.render(label, 1, colours[i])
		for i, label in enumerate(labels)]

	def on_draw():
		screen.fill(0x000000)

		for ii, iterator in enumerate(points):
			if len(iterator) > 1:
				f = lambda x: int(width/len(iterator)*x)
				g = int
				for i in xrange(len(iterator)-1):
					pygame.draw.line(screen, colours[ii],
						(f(i), g(iterator[i]*height)),
						(f(i+1), g(iterator[i+1]*height)))
				z = font.render("{:.02%}".format(iterator[-1]), 1, colours[ii])
				screen.blit(z, (width//2, height-30-40*ii))
			screen.blit(labels[ii], (10, height-30-40*ii))

		pygame.display.flip()

	def update():
		for i in xrange(len(iterators)):
			try:
				n = iterators[i].next()
				points[i].append(n)
			except StopIteration:
				pass

	while True:
		for e in pygame.event.get():
			if e.type == pygame.locals.QUIT:
				pygame.quit()
		update()
		on_draw()

def plot_over_iterator(iterator):
	"""
	Plot values as yielded by an iterator.

	iterator: An iterator which yields percentages.
	"""

	return plot_over_iterators([iterator])