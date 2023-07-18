import sys
import math
import numpy as np
import numpy.ma as ma
from cellpose import plot, utils
from matplotlib import pyplot as plt

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

if __name__ == '__main__':
    f = open("nucleiMaskLocation.txt")
    nucDat = np.load(f.readline().rstrip(), allow_pickle=True).item()
    f.close()

    f = open("cytoMaskLocation.txt")
    cytoDat = np.load(f.readline().rstrip(), allow_pickle=True).item()
    f.close()

    # plot image with outlines overlaid in red
    nucOutlines = utils.outlines_list(nucDat['masks'])
    cytoOutlines = utils.outlines_list(cytoDat['masks'])

    f = open("imgLocation.txt")
    samplingImage = plt.imread(f.readline().rstrip())
    f.close()

    masks = []
    nucWholeMask = nucDat['masks']
    nucWholeMask = nucWholeMask > 0

    cytoWholeMask = cytoDat['masks']

    # i = 0
    for o in nucOutlines:
        # nuclei
        # get average x and y
        centerX = 0
        centerY = 0
        minX = o[:,0][0]
        maxX = o[:,0][0]

        plt.plot(o[:,0], o[:,1], color='r')
        # plt.plot(c[:,0], c[:,1], color='r')

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

        # cytoplasm
        # see if there is a cytoplasm that is close enough to a nucleus to use
        hasCloseCytoplasm = False
        closeMaskId = 1
        for c in cytoOutlines: 
            cytoCenterX = 0
            cytoCenterY = 0
            for x in c[:,0]:
                cytoCenterX+=x
            for y in c[:,1]:
                cytoCenterY += y
            cytoCenterX /= len(c[:,0])
            cytoCenterY /= len(c[:,1])
            if math.dist([centerX, centerY], [cytoCenterX, cytoCenterY]) < 50:
                hasCloseCytoplasm = True
                break
            closeMaskId+=1

        # use only the relavant part of the cytoplasm mask 
        mask = cytoWholeMask == closeMaskId
        # use original circle method if there are no valid cytoplasm masks
        if not hasCloseCytoplasm:
            plt.plot(centerX, centerY, marker=".", markerfacecolor=(0, 0, 0, 0), markeredgecolor=(0, 0, 1, 1), markersize=2*stdMax)
            h, w = samplingImage.shape[:2]
            mask = create_circular_mask(h, w, center=(centerX, centerY), radius=2*stdMax)
        
        # remove the nucleus from the mask
        mask[nucWholeMask] = 0
        masks.append(mask)
        # i+=1

    # stitch together all of the masks
    fullMask = np.zeros(nucWholeMask.shape)
    for mask in masks:
        fullMask = np.add(fullMask, mask)
    fullMask = fullMask > 0

    samplingImage[~fullMask] = 0
    plt.imshow(samplingImage)
    plt.show()

if __name__ == '__main__':
    multiprocessing.freeze_support()
