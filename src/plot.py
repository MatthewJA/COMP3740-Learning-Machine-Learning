#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot a value over time as it is calculated.
"""

from __future__ import division

from random import randrange

import pyglet

def plot_over_iterators(iterators, labels=()):
	"""
	Plots values as yielded by iterators.

	iterator: A list of iterators which yield percentages.
	"""

	width = height = 500

	window = pyglet.window.Window(width, height)

	points = [[] for i in xrange(len(iterators))]
	colours = [(randrange(256), randrange(256), randrange(256))
		for i in xrange(len(iterators))]
	labels = [
		pyglet.text.Label(label,
                          font_name='Arial',
                          font_size=20,
                          x=10, y=height-10-i*30,
                          anchor_x="left", anchor_y="top",
                          color=colours[i]+(255,))
		for i, label in enumerate(labels)]

	@window.event
	def on_draw():
		window.clear()

		for label in labels:
			label.draw()

		for ii, iterator in enumerate(points):
			if len(iterator) > 1:
				f = lambda x: int(width/len(iterator)*x)
				g = int
				scaled = [(f(i), g(val*height))
					for i, val in enumerate(iterator)]
				scaled = [s for p in scaled for s in p]
				pyglet.graphics.draw(len(iterator), pyglet.gl.GL_LINE_STRIP,
					("v2i", scaled),
					("c3B", colours[ii]*len(iterator)))

	def update(dt):
		for i in xrange(len(iterators)):
			try:
				n = iterators[i].next()
				points[i].append(n)
			except StopIteration:
				pass

	pyglet.clock.schedule(update)
	pyglet.app.run()

def plot_over_iterator(iterator):
	"""
	Plot values as yielded by an iterator.

	iterator: An iterator which yields percentages.
	"""

	return plot_over_iterators([iterator])