#!/usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import division

import cPickle
import gzip
import os
import sys
import time
import numpy as np
import Tkinter

"""
Classify data by comparing an input to an average vector for each label.

Principal Author: Buck Shlegeris
"""


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

def classify_by_average(things, thing_to_classify):
    # this gets 37.090000% wrong
    vals = np.matrix([thing_to_classify]) * things.transpose()

    return np.argmax(vals)

# def

def classify_by_neighbours(data, labels, item, k=10):
    # with k=10, this gets 24% wrong and takes 80 seconds
    import heapq

    inner_products = data.dot(np.array(item).transpose())

    thing = zip(inner_products.flatten(), labels)

    heapq.heapify(thing)

    voters = [x[1] for x in heapq.nlargest(k, thing)]

    from itertools import groupby as g
    return max(g(sorted(voters)), key= lambda (x, v):(len(list(v)),-voters.index(x)))[0]

# 1*784 * 784*10 = 1*10


def guess_with_average_vector(train_set, test_set):
    images, labels = train_set
    print "calculating averages..."

    things = np.array([get_nth_average(images, labels, x) for x in range(10)])
    print "things shape = ", things.shape

    labels = test_set[1][0:10000]
    guesses = [classify_by_average(things, test_set[0][x]) for x in range(10000)]

    accuracy = sum(x != y for (x,y) in zip(labels, guesses)) / 10000.0

    print "%f%% were wrong"%(accuracy * 100)

def guess_with_neighbours(train_set, test_set, k=10):
    guesses = []
    for (index, x) in enumerate(test_set[0]):
        guesses.append(classify_by_neighbours(train_set[0], train_set[1], x, k))

    labels = test_set[1]

    accuracy = sum(x != y for (x,y) in zip(labels, guesses)) / len(labels)

    print "%f%% were wrong"%(accuracy * 100)

    return accuracy

def guess_with_all_neighbours_2(train_set, test_set):
    give(up)

if __name__ == '__main__':
    print "loading..."

    train_set, valid_set, test_set = load_data("../../DeepLearningTutorials/data/mnist.pkl.gz")

    print "what"

    accuracies = {x:guess_with_neighbours(train_set, [test_set[0][0:1000],test_set[1][0:1000]], x)
                        for x in [19,20,21,22]}

    print accuracies

    # view_image(images[0], 28, 28)


