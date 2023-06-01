import pandas as pd
import xml.etree.ElementTree as ET

def parse_XML(xml_file,category):
    """Parse the input XML file and store the result in a pandas
    DataFrame
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
