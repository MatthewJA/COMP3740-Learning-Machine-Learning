#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
Dodgy way to view matrices as images.

Principal Author: Matthew Alger
"""

from __future__ import division

import numpy

def view_real_images(matrix, width=28, height=28):
    # Name courtesy of PHP
    import pygame
    import pygame.locals
    import sys

    pygame.init()
    screen = pygame.display.set_mode((width+100, height+100))
    clock = pygame.time.Clock()

    ii = 0
    while True:
        for e in pygame.event.get():
            if e.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()

        m = numpy.amax(matrix[:,ii])

        pixelArray = pygame.PixelArray(screen)
        for i in xrange(width*height):
                x = i % width
                y = i // width
                pixel = matrix[y*width + x, ii]/m
                pixelArray[x+50, y+50] = 0x1000000*int(pixel*255) + 0x10000*int(pixel*255) + 0x100*int(pixel*255) + 0xFF
        del pixelArray

        if ii < matrix.shape[1] - 1:
            ii += 1

        pygame.display.flip()
        clock.tick(1)