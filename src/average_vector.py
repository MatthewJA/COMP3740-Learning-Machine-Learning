from __future__ import division

import cPickle
import gzip
import os
import sys
import time
import numpy as np
import Tkinter


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return train_set, valid_set, test_set


def get_nth_average(images, labels, label):
    # this is stupid and can be optimised.
    # However, it's not annoyingly slow, so yolo

    print "starting collection..."
    slow_list = []
    for (index, thing) in enumerate(labels):
        if thing == label:
            slow_list.append(images[index])

    these_images = np.array(slow_list)
    print "done"

    return np.mean(these_images, axis=0)

def classify(things, thing_to_classify):
    vals = np.matrix([thing_to_classify]) * things.transpose()

    return np.argmax(vals)

# 1*784 * 784*10 = 1*10

def view_real_images(images, labels, width=28, height=28):
    # Name courtesy of PHP
    import pygame
    import pygame.locals
    import sys

    pygame.init()
    screen = pygame.display.set_mode((width+100, height+100))
    clock = pygame.time.Clock()

    ii = 0
    while True:
        print labels[ii]
        for e in pygame.event.get():
            if e.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()

        pixelArray = pygame.PixelArray(screen)
        for i in xrange(width*height):
                x = i % width
                y = i // width
                pixel = images[ii][y*width + x]
                pixelArray[x+50, y+50] = 0x1000000*int(pixel*255) + 0x10000*int(pixel*255) + 0x100*int(pixel*255) + 0xFF
        del pixelArray

        if ii < len(images)-1:
            ii += 1

        pygame.display.flip()
        clock.tick(1)



if __name__ == '__main__':
    print "loading..."

    train_set, valid_set, test_set = load_data("../../DeepLearningTutorials/data/mnist.pkl.gz")

    images, labels = train_set

    # view_image(images[0], 28, 28)

    print "calculating averages..."

    things = np.array([get_nth_average(images, labels, x) for x in range(10)])
    print "things shape = ", things.shape

    labels = test_set[1][0:10000]
    guesses = np.array([classify(things, test_set[0][x]) for x in range(10000)])

    accuracy = sum(x != y for (x,y) in zip(labels, guesses)) / 10000.0

    print "%f%% were wrong"%(accuracy * 100)
