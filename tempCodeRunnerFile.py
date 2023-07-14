import numpy as np
import multiprocessing
from cellpose import plot, utils
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.path import Path

if __name__ == '__main__':
    def create_circular_mask(h, w, center=None, radius=None):
        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask

    f = open("maskLocation.txt")
    dat = np.load(f.readline().rstrip(), allow_pickle=True).item()
    f.close()
    # plot image with masks overlaid
    # mask_RGB = plot.mask_overlay(dat['img'], dat['masks'],
    #                         colors=np.array(dat['colors']))

    # plot image with outlines overlaid in red
    outlines = utils.outlines_list(dat['masks'])

    f = open("imgLocation.txt")
    samplingImage = plt.imread(f.readline().rstrip())
    f.close()

    masks = []
    wholeMask = dat['masks']
    wholeMask = wholeMask > 0
    # print(wholeMask.shape)
    # print(np.max(wholeMask))
    # print(wholeMask[1234,243])
    i = 0
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

        #plt.plot(centerX, centerY, color='r', marker=".")

        plt.plot(centerX, centerY, marker=".", markerfacecolor=(0, 0, 0, 0), markeredgecolor=(0, 0, 1, 1), markersize=2*stdMax)

        h, w = samplingImage.shape[:2]
        mask = create_circular_mask(h, w, center=(centerX, centerY), radius=2*stdMax)
        mask[wholeMask] = 0
        masks.append(mask)

    fullMask = np.zeros(wholeMask.shape)
    for mask in masks:
        fullMask = np.add(fullMask, mask)
    fullMask = fullMask > 0

    samplingImage[~fullMask] = 0
    plt.imshow(samplingImage)
    plt.show()