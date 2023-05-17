from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from cv2 import ximgproc
import higra as hg

from filter import *
from segment_graph import *
import time
from bndbox import *

try:
    from utils import * # imshow, locate_resource, get_sed_model_file
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())

# --------------------------------------------------------------------------------
# Segment an image:
# Returns a color image representing the segmentation.
#
# Inputs:
#           input_path: path of image to segment.
#           sigma: to smooth the image.
#           k: constant for threshold function.
#           min_size: minimum component size (enforced by post-processing stage).
#
# Returns:
#        u : the universe object created
#        bb : the bndbox dictionnary generated
# --------------------------------------------------------------------------------
def segment_felzenszwalb(input_path, sigma, k, min_size,gt_path=""):
    in_image = plt.imread(input_path)

    start_time = time.time()
    height, width, band = in_image.shape
    print("Height:  " + str(height))
    print("Width:   " + str(width))
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

    # boundingbox
    bb = BndBox(np.unique(u.elts[:,2]),width,height)

    for y in range(height):
        for x in range(width):
            pixel_id = y * width + x

            comp = u.find(pixel_id)
            output[y, x, :] = colors[comp, :]

            # check if actual pixel is an outline of his seg bndbox
            bb.check_pixel(str(comp),pixel_id)
            
    elapsed_time = time.time() - start_time
    print(
        "Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
            int(elapsed_time % 60)) + " seconds")

    # ground thruth path 
    if gt_path == "":
        gt_path = "/".join(input_path.split('/')[:2]) + "/Annotations/" + input_path.split('/')[-1].rstrip(".jpg") + ".xml"
    
    # init dict and dataframe to eval bndbox & gt
    bb.init_eval(gt_path)

    # start eval bndbox from gt
    bb.start_eval(verbose=False)

    # displaying the result
    fig = plt.figure()
    #plt.title(f"sigma = {sigma} k = {k} min_size = {min_size}")
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(in_image)
    a.set_title('Original Image')

    for comp in bb.get_bndbox_id():
        pt = bb.get_bndbox(comp)

        rect = patches.Rectangle((2+(pt[0][0]%width), 2+(pt[1][0]/width)),
                                 ((pt[0][1]%width)-2) - (pt[0][0]%width)+2,
                                 ((pt[1][1]/width)-2) - (pt[1][0]/width),
                linewidth=1.5, edgecolor=bb.get_bndbox_color(comp), facecolor='none')

        a.add_patch(rect)

    a = fig.add_subplot(1, 2, 2)
    plt.imshow(output.astype('uint8'))
    a.set_title('Segmented Image')

    fig.savefig(f"result/{input_path.split('/')[-1]}")

    plt.show()

    return u, bb

def segment_watershed(input_path,n_comp=9,gt_path=""):
    in_image = plt.imread(input_path)
    
    # switch to float to avoir numerical issue with uint8
    in_image = in_image.astype(np.float32)/255
    
    start_time = time.time()
    height, width, band = in_image.shape
    print("Height:  " + str(height))
    print("Width:   " + str(width))

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

    elapsed_time = time.time() - start_time
    print(
        "Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
            int(elapsed_time % 60)) + " seconds")

    # ground thruth path 
    if gt_path == "":
        gt_path = "/".join(input_path.split('/')[:2]) + "/Annotations/" + input_path.split('/')[-1].rstrip(".jpg") + ".xml"
    
    # init dict and dataframe to eval bndbox & gt
    bb.init_eval(gt_path)

    # start eval bndbox from gt
    bb.start_eval(verbose=True)

    # displaying the result
    fig = plt.figure()

    a = fig.add_subplot(1, 2, 1)
    plt.imshow(in_image)
    a.set_title('Original Image')

    for comp in bb.get_bndbox_id():
        pt = bb.get_bndbox(comp)

        rect = patches.Rectangle((2+(pt[0][0]%width), 2+(pt[1][0]/width)),
                                 ((pt[0][1]%width)-2) - (pt[0][0]%width)+2,
                                 ((pt[1][1]/width)-2) - (pt[1][0]/width),
                linewidth=1.5, edgecolor=bb.get_bndbox_color(comp), facecolor='none')

        a.add_patch(rect)

    a = fig.add_subplot(1, 2, 2)
    plt.imshow(label_watershed)
    a.set_title('Segmented Image')

    fig.savefig(f"result/watershed_{input_path.split('/')[-1]}")

    plt.show()

    return graph_saliency, bb


if __name__ == "__main__":
    import sys

    sigma = 0.5
    k = 500
    min = 50

    input_path = sys.argv[1]

    # Loading the image

    print("Loading is done.")
    print("processing...")

    segment_felzenszwalb(input_path, sigma, k, min)
