import numpy as np
import multiprocessing
import utils

from cellpose import plot, utils
from matplotlib import pyplot as plt
dat = np.load('/Users/genechang/Downloads/TrainingData01/0_seg.npy', allow_pickle=True).item()



if __name__ == '__main__':
    #m akes sure code doesn't make 10000000 instances of python on your computer
    multiprocessing.freeze_support()

    # plot image with masks overlaid
    mask_RGB = plot.mask_overlay(dat['img'], dat['masks'],
                            colors=np.array(dat['colors']))

    # plot image with outlines overlaid in red
    outlines = utils.outlines_list(dat['masks'])
    plt.imshow(dat['img'])
    for o in outlines:
        CenterX = 0
        CenterY = 0
        plt.plot(o[:,0], o[:,1], color='r')
        for x in o[:,0]:
            CenterX += x
        for y in o[:,1]:
            CenterY += y 
        
    plt.show()

    if __name__ == '__main__':
        multiprocessing.freeze_support()

