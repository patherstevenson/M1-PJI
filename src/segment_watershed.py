#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`Segment Watershed` module
:author: Pather Stevenson - Faculté des Sciences et Technologies - Univ. Lille <http://portail.fil.univ-lille1.fr>_
:date: May 2023

Segment Watershed Module

:doc: <https://higra.readthedocs.io/en/stable/notebooks.html>

"""

import numpy as np
from cv2 import ximgproc
import higra as hg

from bndbox import * 

try:
    from utils import * # imshow, locate_resource, get_sed_model_file
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())

def segment_watershed(in_image: np.array,height: int,width: int,n_comp=9) -> tuple:
    """
    Perform a watershed segmentation on the given image and
    retain exactly the given number of larger regions to retain in the hierachy

    <https://hal.science/hal-01344727/document>

    :param in_image: The image data as array
    :param height: the height of in_image
    :param width: the width of in_image
    :param n_comp: number of larger regions to retain in the hierachy
    :type in_image: numpy.array
    :type height: int
    :type width: int
    :type n_comp: int

    :return: the array indicating which region each pixel belongs to and the associated BndBox object of this segmentation
    :rtype: tuple (numpy.ndarray, BndBox)
    """
    # get gradient image 
    detector = ximgproc.createStructuredEdgeDetection(get_sed_model_file())
    gradient_image = detector.detectEdges(in_image)

    # contruct an edge weighted graph, and transfer gradient to edge weights
    graph = hg.get_4_adjacency_graph(in_image.shape[:2])
    edge_weights = hg.weight_graph(graph, gradient_image, hg.WeightFunction.mean)

    # watershed hierarchy by area
    tree, altitudes = hg.watershed_hierarchy_by_area(graph, edge_weights)
    #output = hg.graph_4_adjacency_2_khalimsky(graph, hg.saliency(tree, altitudes))**0.5

    # saillence graph
    graph_saliency = hg.saliency(tree, altitudes)

    # get all index of pixel which belong to comp which are < to the n_comp th highest comp
    if n_comp < len(np.unique(graph_saliency)):
        index = graph_saliency < np.unique(graph_saliency)[-n_comp]

        # replace all index by a weight of 0 (= ignoring them)
        graph_saliency[index] = 0

    # get pixel label (= comp) from adj graph and saliency graph
    label_watershed = hg.labelisation_watershed(graph,graph_saliency) 

    # bindingbox
    bb = BndBox(np.unique(label_watershed),width,height)

    # calculate bb from watershed seg
    for y in range(height):
        for x in range(width):
            bb.check_pixel(str(label_watershed[y][x]),y*width+x)

    return label_watershed, bb
