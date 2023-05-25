import numpy as np
from cv2 import ximgproc
import higra as hg

from bndbox import * 

try:
    from utils import * # imshow, locate_resource, get_sed_model_file
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())

def segment_watershed(in_image,height,width,n_comp=9):
    
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
