#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`bndbox` module
:author: Pather Stevenson - Faculté des Sciences et Technologies - Univ. Lille <http://portail.fil.univ-lille1.fr>_
:date: May 2023

bndbox Module

"""

import numpy as np
import pandas as pd
from xml_parser import parse_XML

class BndBox:
    """
    Create a BndBox object which can calculate and evaluate bounding boxes
    from segmented image and given ground truths
    """
    def __init__(self,label: tuple, w: int,h: int):
        """
        Create a BndBox object which can calculate and evaluate bounding boxes
        from segmented image and given ground truths.

        :param label:
        :param w:
        :param h:

        :type label:
        :type w:
        :type h:

        :UC: type(w) == type(h) == int
        :UC: 0 < w and 0 < h     
        """
        self.bndbox = {str(comp) : np.array([[w-1,0],[w*h-1,0]]) for comp in label}
        self.overlap_05 = {}
        self.max_overlap = {}
        self.abo = {}
        self.bndbox_color = {}
        self.df_bndbox = pd.DataFrame(columns=['name','xmin','ymin','xmax','ymax'])
        self.w = w
        self.h = h

    def get_bndbox(self,comp: str) -> np.array:
        """
        Return for a given region id the array of pixel id of each ends,

        format : [[most left, most right],
                  [most up, most down]]

        :param comp: id of the
        :type comp: str

        :return: array of pixel id of each ends of the given region id
        :rtype: np.array

        :UC: type(comp) == str
        """
        return self.bndbox[comp]
    
    def get_bndbox_id(self):
        """
        Return the bounding box id of each regions

        :return: bndbox id of each regions
        :rtype: dict_keys()

        :UC: None
        """
        return self.bndbox.keys()
    
    def get_nb_bndbox(self) -> int:
        """
        Return the number of bounding box / regions 

        :return: number of regions
        :rtype: int

        :UC: None
        """
        return len(list(self.get_bndbox_id()))
    
    def get_bndbox_color(self,comp : str) -> str:
        """
        Return the assign color for the given bounding box id, 
        red for bounding box without max overlap with some ground thruth
        otherwise green

        :param comp: id of the bounding box
        :type comp: str

        :return: assigned color for the given bounding box id (red or green)
        :rtype: str

        :UC: comp must be in self.bndbox.keys()
        """
        return self.bndbox_color[comp]

    def init_eval(self,gt_path: str,category: str) -> None:
        """
        Init dictionaries in order to perform the evaluation phase

        :param gt_path: path of groundtruth file
        :param category: category name of objects to detect in images
        :type gt_path: str
        :type category: str

        :return: None
        :rtype: None

        :UC: None
        """
        self.df_bndbox = parse_XML(gt_path,category) # dataframe of the given category groundthruth
        self.overlap_05 = {i : ('None',0) for i in range(self.df_bndbox.shape[0])} # all overlap > 0.5
        self.max_overlap = {i : ('None',0) for i in range(self.df_bndbox.shape[0])} # max overlap > 0.5
        self.abo = {name : 0 for name in np.unique(self.df_bndbox["name"].values)} # ABO for each groundtruth of category
        self.bndbox_color = {comp : "r" for comp in list(self.get_bndbox_id())} # color assigned to each bounding box

    def check_pixel(self,comp: str,pixel_id: int) -> None:
        """
        Verify if a given pixel id is a the most left, most right, most up or most down pixel of his region which is given
        
        :param comp: region id
        :param pixel_id: pixel id to check
        :type comp: str
        :type pixel_id: int

        :return: None
        :rtype: None
        """
        # most left
        if self.bndbox[comp][0][0]%self.w > pixel_id%self.w: 
            self.bndbox[comp][0][0] = pixel_id

        # most right
        if self.bndbox[comp][0][1]%self.w < pixel_id%self.w:
            self.bndbox[comp][0][1] = pixel_id

        # most up
        if self.bndbox[comp][1][0] > pixel_id: 
            self.bndbox[comp][1][0] = pixel_id
        
        # most down
        if self.bndbox[comp][1][1] < pixel_id:
            self.bndbox[comp][1][1] = pixel_id

    def overlap(self,l1, r1, l2, r2):
        """
        Calculate the total overlapping area of two given rectangle represented by
        bottom left and top right points

        :param l1: bottom left point (x,y) of the first rectangle
        :param r1: top right point (x,y) of the first rectangle
        :param: l2: bottom left point (x,y) of the second rectangle
        :param r2: top right point (x,y) of the second rectangle

        :type l1: Tuple
        :type r1: Tuple
        :type l2: Tuple
        :type r2: Tuple

        :return: the total overlapping area from the two given rectangles, return 0 they don't overlap
        :rtype: int
        """
        x = 0
        y = 1
        
        area1 = np.abs(l1[x] - r1[x]) * np.abs(l1[y] - r1[y])
        area2 = np.abs(l2[x] - r2[x]) * np.abs(l2[y] - r2[y])
        
        x_dist = np.min([r1[x], r2[x]]) - np.max([l1[x], l2[x]])
        y_dist = np.min([l1[y], l2[y]]) - np.max([r1[y], r2[y]])
        
        areaI = 0
        
        if x_dist > 0 and y_dist > 0:
            areaI = x_dist * y_dist
        else:
            return 0
        
        return areaI/(area1 + area2 - areaI)

    def _eval_abo(self) -> None:
        """
        Evalute and save ABO for each groundtruth in self.abo dict

        :return: None
        :rtype: None
        """
        l_label = np.unique(self.df_bndbox['name'])

        for label in l_label:
            index = (self.df_bndbox['name'] == label).values

            self.abo[label] = (1/len(index)) * np.sum(self.max_overlap[i][1] for i in np.arange(0,len(self.max_overlap.keys()),1)[index])  

    def start_eval(self,verbose=False) -> None:
        """
        Launch the evaluation phase of quality of bounding boxes calculated
        compared with ground truths by using the measure of average overlap
        
        :param verbose: verbosity
        :type verbose: bool
        
        :return: None
        :rtype: None
        """
        for i in range(self.df_bndbox.shape[0]):
            for comp in self.get_bndbox_id():
    
                l1 = (int(self.get_bndbox(comp)[0][0]%self.w),int(self.get_bndbox(comp)[1][1]/self.w))
                r1 = (int(self.get_bndbox(comp)[0][1]%self.w),int(self.get_bndbox(comp)[1][0]/self.w))
                l2 = (self.df_bndbox.iloc[:,1:]["xmin"][i],self.df_bndbox.iloc[:,1:]["ymax"][i])
                r2 = (self.df_bndbox.iloc[:,1:]["xmax"][i],self.df_bndbox.iloc[:,1:]["ymin"][i])
        
                tmp_overlap = self.overlap(l1,r1,l2,r2)

                if verbose: print(i,comp,tmp_overlap,self.overlap_05[i][1])
                
                if tmp_overlap > 0.5 and tmp_overlap > self.overlap_05[i][1]:
                    self.bndbox_color[self.overlap_05[i][0]] = "r"
                    self.overlap_05[i] = (comp,tmp_overlap)
                    self.bndbox_color[comp] = "g"
                if tmp_overlap > self.max_overlap[i][1]:
                    self.max_overlap[i] = (comp,tmp_overlap)
        
        # calculate ABO for each category of groundtruth
        self._eval_abo()