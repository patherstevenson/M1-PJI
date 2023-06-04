#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`Main` module
:author: Pather Stevenson - Faculté des Sciences et Technologies - Univ. Lille <http://portail.fil.univ-lille1.fr>_
:date: May 2023

Main Module
"""

import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from bndbox import BndBox
from segment_felzenszwalb import segment_felzenszwalb
from segment_watershed import segment_watershed

def usage():
    print("USAGE\n\n- Felzenszwalb :\n\n\t$ python main.py [input_path] f [category] [gt_path]\n\n- Watershed:\n\n\t$ python main.py [input_path] w [category] [n_comp] [gt_path]\n")
    print("input_path : image path to segment (ex: ../data/VOC2012_train_val/JPEGImages/person/XXXX.jpg)")
    print("category   : category of bounding box (ex: person, cat, bicycle, chair, ...)")
    print("n_comp     : only for watershed method, number of larger regions to retain in the hierachy, default 9 for the 10 most larger regions")
    print("gt_path    : path of the associated groundtruth wit the given input image, optional if you use VOC2012 dataset file tree (../data/VOC2012_train_val/Annotations/category/XXXX.xml)")

def plot_segment(in_image: np.ndarray,input_path: str,output: np.ndarray,bb: BndBox,category: str,k="",method="felzenszwalb",save=True) -> None:
    """
    
    Plot and save (in ../result/category/...) the original image with the red and green bounding box calculated from
    the given segmentation method, and the segmented image

    :param in_image: The image data as array
    :param input_path: path of the original image
    :param output: the segmented image
    :param bb: BndBox object which contains the dict of bounding box calculated with assigned colors
    :param category: name of the category of the given in_image
    :param k: threshold constant
    :param method: indicate the segmentation method used : felzenszwalb or watershed
    :param save: True to save the plot, otherwise False

    :type in_image: numpy.ndarray
    :type input_path: str
    :type output: numpy.ndarray
    :type bb: BndBox
    :type category: str
    :type k: str
    :type method: str
    :type save: bool

    :return: None
    :rtype: None
    """
    fig = plt.figure()

    a = fig.add_subplot(1, 2, 1)
    plt.imshow(in_image)
    a.set_title('Original Image')

    for comp in bb.get_bndbox_id():
        pt = bb.get_bndbox(comp)

        rect = patches.Rectangle((2+(pt[0][0]%bb.w), 2+(pt[1][0]/bb.w)),
                                 ((pt[0][1]%bb.w)-2) - (pt[0][0]%bb.w)+2,
                                 ((pt[1][1]/bb.w)-2) - (pt[1][0]/bb.w),
                linewidth=1.5, edgecolor=bb.get_bndbox_color(comp), facecolor='none')

        a.add_patch(rect)

    a = fig.add_subplot(1, 2, 2)
    plt.imshow(output.astype('uint8'))
    a.set_title(f'{method} segmentation'.capitalize())
    if k != "": k = str(k) + "_"
    if save: fig.savefig(f"../result/{category}/{method}_{k}{input_path.split('/')[-1]}")
    
    plt.show()

def segmentation(input_path: str,method="felzenszwalb",kwargs={"sigma" : 0.5, "k" : 500, "min_size" : 50},n_comp=9,category="person",gt_path="",save=True,verbose=False) -> tuple:
    """

    Perform the segmentation method given (felzenszwalb or watershed) on the given input image path
    by using specified parameters (kwargs for felzenszwalb, n_comp for watershed).
    Then calculate bounding box and evalute their quality for the given category
    by measuring the ABO with the given associated groundtruth

    :param input_path: path of the image to segment
    :param method: segmentation method to use, must be felzenszwalb or watershed
    :param kwargs: only for felzenszwalb method, dictionnary with sigma (for gaussian filter), k (threshold function), min_size (minimum component size)
    :param n_comp: only for watershed method, number of larger regions to retain in the hierachy
    :param category: category name of the given image
    :param gt_path: path of the associated groundtruth wit the given input image
    :param save: True to save the result, otherwise False
    :param verbose: verbosity

    :type input_path: str
    :type method: str
    :type kwargs: dict
    :type n_comp: int
    :type category: str
    :type gt_path: str
    :type save: bool
    :type verbose: bool

    :return: the BndBox object associated with the segmentation, the segmented image which allows to identify which region each pixel belongs to
    :rtype: tuple (BndBox, numpy.ndarray)

    """
    assert(method in ["felzenszwalb", "watershed"])

    in_image = plt.imread(input_path)

    height, width, band = in_image.shape
    
    if verbose : print("Height:  " + str(height),"\nWidth:   " + str(width),end="\n")

    start_time = time.time()
    
    # get output & bndbox from the segmentation used
    if method == "felzenszwalb":
        assert(band == 3)
        output, bb = segment_felzenszwalb(in_image,**kwargs,height=height,width=width)
    else:
        # switch to float to avoid numerical issue with uint8
        in_image = in_image.astype(np.float32)/255
        output, bb = segment_watershed(in_image,height,width,n_comp=n_comp)

    elapsed_time = time.time() - start_time

    if verbose : print("Execution time: " + str(int(elapsed_time / 60))
                       + " minute(s) and " + str(int(elapsed_time % 60))
                       + " seconds",end="\n\n")

    # ground thruth xml path
    if gt_path == "": gt_path = "/".join(input_path.split('/')[:3]) + "/Annotations/" + category + "/" + input_path.split('/')[-1].rstrip(".jpg") + ".xml"
    
    # init dict and dataframe to eval bndbox & gt
    bb.init_eval(gt_path,category)

    # start eval bndbox from gt
    bb.start_eval(verbose=False) # verbose=verbose/True to show all calculated overlap

    # plot & save results
    plot_segment(in_image,input_path,output,bb,category,k=kwargs['k'],method=method,save=save)

    return bb, output

if __name__ == "__main__":
    import sys
    len_argv = len(sys.argv)
    
    if (len_argv >= 4):

        input_path = sys.argv[1]
        method = sys.argv[2]
        category = sys.argv[3]
        gt_path = "" # last in argv
        verbose = True

        if method == "f": # felzenszwalb 
            if len_argv == 5:
                gt_path = sys.argv[4]

            segmentation(input_path,method="felzenszwalb",category=category,gt_path=gt_path,save=False,verbose=verbose)

        elif method == "w": # watershed
            n_comp = 9
            if len_argv > 4:
                n_comp = sys.argv[4]
            if len_argv > 5:
                gt_path = sys.argv[5]

            segmentation(input_path,method="watershed",category=category,n_comp=n_comp,gt_path=gt_path,save=False,verbose=verbose)
        
        else:
            sys.exit(f"Method unknow : {method}",end='\n\n')
    else:
        usage()
        exit()
