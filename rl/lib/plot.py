#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot a value over time as it is calculated.

Principal Author: Matthew Alger
"""

from __future__ import division

import time
from random import randrange

import pygame
import pygame.locals

def plot_over_iterators(iterators, labels=(), scale=1):
	"""
	Plots values as yielded by iterators.

	iterator: A list of iterators which yield percentages.
	labels: A tuple of names for each iterator.
	scale: y-scale factor.
	"""

	pygame.init()

	info = pygame.display.Info()
	width = info.current_w
	height = info.current_h

	screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
	font = pygame.font.SysFont("sans-serif", 30)

	points = [[] for i in xrange(len(iterators))]
	colours = [(randrange(128, 256), randrange(128, 256), randrange(128, 256))
		for i in xrange(len(iterators))]
	labels = [font.render(label, 1, colours[i])
		for i, label in enumerate(labels)]
	
	saved = False

	def on_draw():
		screen.fill(0x000000)

		for ii, iterator in enumerate(points):
			if len(iterator) > 1:
				f = lambda x: int(width/len(iterator)*x)
				g = int
				for i in xrange(len(iterator)-1):
					pygame.draw.line(screen, colours[ii],
						(f(i), g(min(height, iterator[i]*height*scale))),
						(f(i+1), g(min(height, iterator[i+1]*height*scale))))
				z = font.render("{:.02%}".format(float(iterator[-1])), 1, colours[ii])
				screen.blit(z, (width//2, height-30-40*ii))
			screen.blit(labels[ii], (10, height-30-40*ii))

		pygame.display.flip()

	def update():
		for i in xrange(len(iterators)):
			try:
				n = iterators[i].next()
				points[i].append(n)
			except StopIteration:
				return False
		return True

	while True:
		for e in pygame.event.get():
			if e.type == pygame.locals.QUIT:
				pygame.quit()
			if e.type == pygame.locals.MOUSEBUTTONDOWN:
				if not saved:
					saved = True
					date = str(time.strftime("%d-%m-%Y--%H:%M:%S"))
					pygame.image.save(screen, "../plots/" + date + ".png")
				pygame.quit()
		success = update()
		if not success and not saved:
			saved = True
			date = str(time.strftime("%d-%m-%Y--%H:%M:%S"))
			pygame.image.save(screen, "../plots/" + date + ".png")
		on_draw()

def plot_over_iterator(iterator):
	"""
	Plot values as yielded by an iterator.

	iterator: An iterator which yields percentages.
	"""

	return plot_over_iterators([iterator])
