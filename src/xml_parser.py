#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`XML Parser` module
:author: Pather Stevenson - Faculté des Sciences et Technologies - Univ. Lille <http://portail.fil.univ-lille1.fr>_
:date: May 2023

XML Parser Module

"""

import pandas as pd
import xml.etree.ElementTree as ET

def parse_XML(xml_file: str,category: str) -> pd.DataFrame:
    """
    Parse the input XML file and store the result in a pandas.DataFrame

    :param xml_file: path of ground truth xml file
    :param category: category of objects to detect
    :type xml_file: str
    :type category: str

    :return: the DataFrame which contains the readed groundtruth of the given category from the given xml file
    :rtype: pandas.DataFrame
    """

    xml_data = open(xml_file, 'r').read() 
    root = ET.XML(xml_data) 

    bndbox = []

    for obj in root.findall("object"):
        if obj.find("name").text == category:
            bndbox.append([obj.find("name").text,
                       int(obj.find("bndbox").find("xmin").text),
                       int(obj.find("bndbox").find("ymin").text),
                       int(obj.find("bndbox").find("xmax").text),
                       int(obj.find("bndbox").find("ymax").text)
                      ])

    out_df = pd.DataFrame(bndbox, columns=["name","xmin","ymin","xmax","ymax"])

    return out_df
