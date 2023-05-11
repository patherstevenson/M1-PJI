import numpy as np

def overlap(l1, r1, l2, r2):
    x = 0
    y = 1
    
    area1 = abs(l1[x] - r1[x]) * abs(l1[y] - r1[y])
    area2 = abs(l2[x] - r2[x]) * abs(l2[y] - r2[y])
    
    x_dist = min(r1[x], r2[x]) - max(l1[x], l2[x])
    
    y_dist = min(r1[y], r2[y]) - max(l1[y], l2[y])
    
    areaI = 0
    
    if x_dist > 0 and y_dist > 0:
        areaI = x_dist * y_dist
    else:
        return 0
        
    return areaI/(area1 + area2 - areaI)

class BndBox:
    def __init__(self,elts,w,h):
        self.bndbox = {str(comp) : np.array([[w-1,0],[w*h-1,0]]) for comp in np.unique(elts[:,2])}
        self.w = w
        self.h = h

    def get_bndbox(self,comp):
        return self.bndbox[comp]
    
    def get_bndbox_id(self):
        return self.bndbox.keys()

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
