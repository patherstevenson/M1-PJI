#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`Segment Felzenszwalb` module
:author: Pather Stevenson - Faculté des Sciences et Technologies - Univ. Lille <http://portail.fil.univ-lille1.fr>_
:date: May 2023

Segment Felzenszwalb Module

:doc: <https://github.com/salaee/pegbis>

"""

import math
import numpy as np
import random

from universe import *
from filter import *
from bndbox import *

def segment_felzenszwalb(in_image: np.ndarray, sigma: float, k: int, min_size: int,height: int,width: int) -> tuple:
    """
    Performs a complete felzenszwalb segmentation and calculate
    bounding box obtained from the segmentation

    <https://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf>

    :param in_image: The image data as array
    :param sigma: value of gaussian filter to smooth the image
    :param k: constant for threshold function
    :param min_size:  minimum component size (enforced by post-processing stage)
    :param height: height of the image to segment
    :param width: width of the image to segment
    :type in_image: numpy.array
    :type sigma: float
    :type k: int
    :type min_size: int
    :type height: int
    :type width: int

    :return: the segmented image and the the associated BndBox object of this segmentation
    :rtype: tuple (numpy.array, BndBox)

    :UC: in_image must be of shape (height,width,3)
    """
    smooth_red_band = smooth(in_image[:, :, 0], sigma)
    smooth_green_band = smooth(in_image[:, :, 1], sigma)
    smooth_blue_band = smooth(in_image[:, :, 2], sigma)

    # build graph
    edges_size = width * height * 4
    edges = np.zeros(shape=(edges_size, 3), dtype=object)
    num = 0
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int(y * width + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y)
                num += 1
            if y < height - 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y + 1) * width + x)
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x, y + 1)
                num += 1

            if (x < width - 1) and (y < height - 2):
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y + 1) * width + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y + 1)
                num += 1

            if (x < width - 1) and (y > 0):
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y - 1) * width + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y - 1)
                num += 1
    # Segment
    u = segment_graph(width * height, num, edges, k)

    # post process small components
    for i in range(num):
        a = u.find(edges[i, 0])
        b = u.find(edges[i, 1])
        if (a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)):
            u.join(a, b)

    num_cc = u.num_sets()
    output = np.zeros(shape=(height, width, 3))

    # pick random colors for each component
    colors = np.zeros(shape=(height * width, 3))
    for i in range(height * width):
        colors[i, :] = random_rgb()

    # bounding box
    bb = BndBox(np.unique(u.elts[:,2]),width,height)

    for y in range(height):
        for x in range(width):
            pixel_id = y * width + x

            comp = u.find(pixel_id)
            output[y, x, :] = colors[comp, :]

            # check if actual pixel is an outline of his seg bndbox
            bb.check_pixel(str(comp),pixel_id)

    return output, bb

def segment_graph(num_vertices: int, num_edges: int, edges: np.ndarray, c: int) -> Universe:
    """
    Returns a disjoint-set forest representing the segmentation

    :param num_vertices: number of vertices in graph
    :param num_edges: number of edges in graph
    :param edges: array of edges
    :param c: constant for threshold function
    :type num_vertices: int
    :type num_edges: int
    :type edges: 
    :type c: int

    :return: a disjoint-set forest representing the segmentation
    :rtype: Universe
    """
    # sort edges by weight (3rd column)
    edges[0:num_edges, :] = edges[edges[0:num_edges, 2].argsort()]
    # make a disjoint-set forest
    u = Universe(num_vertices)
    # init thresholds
    threshold = np.zeros(shape=num_vertices, dtype=float)
    for i in range(num_vertices):
        threshold[i] = get_threshold(1, c)

    # for each edge, in non-decreasing weight order...
    for i in range(num_edges):
        pedge = edges[i, :]

        # components connected by this edge
        a = u.find(pedge[0])
        b = u.find(pedge[1])
        if a != b:
            if (pedge[2] <= threshold[a]) and (pedge[2] <= threshold[b]):
                u.join(a, b)
                a = u.find(a)
                threshold[a] = pedge[2] + get_threshold(u.size(a), c)

    return u


def get_threshold(size: int, c: int) -> float:
    """
    Return the threshold in function of size and c

    :param size: size of component
    :param c: threshold constant

    :return: result of threshold check
    :rtype: float
    """
    return c / size


# returns square of a number
def square(value):
    """
    Return the square of the given value

    :param value: value to square
    :type value: int or float
    """
    return value * value


# randomly creates RGB
def random_rgb():
    """
    Return a random 3-RBG colors channel

    :return: a 3-RBG colors channel
    :rtype: numpy.array
    """
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(0, 255)
    rgb[1] = random.randint(0, 255)
    rgb[2] = random.randint(0, 255)
    return rgb


# dissimilarity measure between pixels
def diff(red_band: np.ndarray, green_band: np.ndarray, blue_band: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """
    Perform the dissimilarity measure between given pixels (x1,y2) and (x2,y2)
    in function of given RBG channels
    
    :param red_brand: red channel
    :param green_band: green channel
    :param blue_band: blue channel
    :type red_brand: numpy.ndarray
    :type green_band: numpy.ndarray
    :type blue_band: numpy.ndarray

    :return: dissimilarity measure between first and second given pixels
    :rtype: float
    """
    result = math.sqrt(
        square(red_band[y1, x1] - red_band[y2, x2]) + square(green_band[y1, x1] - green_band[y2, x2]) + square(
            blue_band[y1, x1] - blue_band[y2, x2]))
    return result