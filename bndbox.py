import numpy as np
from xml_parser import *

def overlap(l1, r1, l2, r2):
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

class BndBox:
    def __init__(self,label,w,h):
        self.bndbox = {str(comp) : np.array([[w-1,0],[w*h-1,0]]) for comp in label}
        self.overlap_05 = {}
        self.bndbox_color = {}
        self.df_bndbox = pd.DataFrame(columns=['name','xmin','ymin','xmax','ymax'])
        self.w = w
        self.h = h

    def get_bndbox(self,comp):
        return self.bndbox[comp]
    
    def get_bndbox_id(self):
        return self.bndbox.keys()
    
    def get_nb_bndbox(self):
        return len(list(self.get_bndbox_id()))
    
    def get_bndbox_color(self,comp):
        return self.bndbox_color[comp]

    def init_eval(self,gt_path):
        self.df_bndbox = parse_XML(gt_path)
        self.overlap_05 = {i : ('None',0) for i in range(self.df_bndbox.shape[0])}
        self.bndbox_color = {comp : "r" for comp in list(self.get_bndbox_id())}

    def check_pixel(self,comp,pixel_id):
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

    def start_eval(self,verbose=False):
        for i in range(self.df_bndbox.shape[0]):
            for comp in self.get_bndbox_id():
    
                l1 = (int(self.get_bndbox(comp)[0][0]%self.w),int(self.get_bndbox(comp)[1][1]/self.w))
                r1 = (int(self.get_bndbox(comp)[0][1]%self.w),int(self.get_bndbox(comp)[1][0]/self.w))
                l2 = (self.df_bndbox.iloc[:,1:]["xmin"][i],self.df_bndbox.iloc[:,1:]["ymax"][i])
                r2 = (self.df_bndbox.iloc[:,1:]["xmax"][i],self.df_bndbox.iloc[:,1:]["ymin"][i])
        
                tmp_overlap = overlap(l1,r1,l2,r2)

                if verbose: print(i,comp,tmp_overlap,self.overlap_05[i][1])
                
                if tmp_overlap > 0.5 and tmp_overlap > self.overlap_05[i][1]:
                    self.bndbox_color[self.overlap_05[i][0]] = "r"
                    self.overlap_05[i] = (comp,tmp_overlap)
                    self.bndbox_color[comp] = "g"
