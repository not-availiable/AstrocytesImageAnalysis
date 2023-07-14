import numpy as np
import multiprocessing
from cellpose import plot, utils
from matplotlib import pyplot as plt




if __name__ == '__main__':
    #m akes sure code doesn't make 10000000 instances of python on your computer
    multiprocessing.freeze_support()

    # plot image with masks overlaid
    def clamp(num, min_value, max_value):
        return max(min(num, max_value), min_value)

    dat = np.load('/Users/genechang/Downloads/TrainingData01/0_seg.npy', allow_pickle=True).item()

    # plot image with masks overlaid
    mask_RGB = plot.mask_overlay(dat['img'], dat['masks'],
                            colors=np.array(dat['colors']))

    # plot image with outlines overlaid in red
    outlines = utils.outlines_list(dat['masks'])

    samplingImage = plt.imread('/Users/genechang/Downloads/TrainingData01/Cell39_Pre.tiff')
    plt.imshow(samplingImage)

    #fig = plt.figure()#fig, axs = plt.subplots(len(outlines))

    for o in outlines:
        #get average x and y
        centerX = 0
        centerY = 0
        minX = o[:,0][0]
        maxX = o[:,0][0]

        plt.plot(o[:,0], o[:,1], color='r')

        for x in o[:,0]:
            centerX+=x
            if x < minX:
                minX = x
            if x > maxX:
                maxX = x
        for y in o[:,1]:
            centerY += y
        centerX /= len(o[:,0])
        centerY /= len(o[:,1])

        #get standard deviation
        stdX = np.std(o[:,0])
        stdY = np.std(o[:,1])
        stdMax = max(stdX, stdY)

        plt.plot(centerX, centerY, color='r', marker=".")
        plt.plot(centerX, centerY, marker=".", markerfacecolor=(0, 0, 0, 0), markeredgecolor=(0, 0, 1, 1), markersize=stdMax)

        #plt.add_patch(patches.Circle((centerX, centerY),radius=stdMax))

        #plt.fill_between([minX, maxX], [o[:,0], o[:,1]], )

    plt.show()

    if __name__ == '__main__':
        multiprocessing.freeze_support()


