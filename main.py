import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from segment_felzenszwalb import segment_felzenszwalb
from segment_watershed import segment_watershed

def plot_segment(in_image,input_path,output,bb,category,k="",method="felzenszwalb",save=True):
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
    if save: fig.savefig(f"result/{category}/{method}_{k}{input_path.split('/')[-1]}")
    
    plt.show()

def segmentation(input_path,method="felzenszwalb",kwargs={"sigma" : 0.5, "k" : 500, "min_size" : 50},n_comp=9,category="person",gt_path="",save=True,verbose=False):
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
    if gt_path == "": gt_path = "/".join(input_path.split('/')[:2]) + "/Annotations/" + category + "/" + input_path.split('/')[-1].rstrip(".jpg") + ".xml"
    
    # init dict and dataframe to eval bndbox & gt
    bb.init_eval(gt_path,category)

    # start eval bndbox from gt
    bb.start_eval(verbose=verbose)

    # plot & save results
    plot_segment(in_image,input_path,output,bb,category,k=kwargs['k'],method=method,save=save)

    return bb, output

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
