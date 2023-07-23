import sys
import os
import math
import numpy as np
import numpy.ma as ma
from cellpose import models, utils
from matplotlib import pyplot as plt
import multiprocessing
import time
import json
from natsort import natsorted

#start timer to measure how long code takes to execute
start_time=time.time()

def load_path(file):
    f = open(file)
    path = f.readline().rstrip()
    f.close()
    return path

def get_center_location(o):
    #takes average
    return o[:, 0].mean(), o[:, 1].mean()
def generate_masks():
    cyto_centers = np.array([get_center_location(c) for c in cytoOutlines])  # convert list of tuples to 2D numpy array
    for o in nucOutlines:
        # nuclei
        # get average x and y
        centerX, centerY = get_center_location(o)
        plt.plot(o[:,0], o[:,1], color='r')

        #get standard deviation
        stdX = np.std(o[:,0])
        stdY = np.std(o[:,1])
        stdMax = max(stdX, stdY)

        # cytoplasm
        distances = np.sqrt(np.sum((cyto_centers - np.array([centerX, centerY]))**2, axis=1))  
        close_cytoplasm_indices = np.where(distances < 50)[0]  # Find indices where distance is less than 50
        if close_cytoplasm_indices.size > 0:
            # there is a cytoplasm that is close enough to a nucleus to use
            # use only the relevant part of the cytoplasm mask 
            mask = cytoWholeMask == close_cytoplasm_indices[0] + 1
        else:
            # use original circle method if there are no valid cytoplasm masks
            plt.plot(centerX, centerY, marker=".", markerfacecolor=(0, 0, 0, 0), markeredgecolor=(0, 0, 1, 1), markersize=2*stdMax)
            h, w = samplingImage.shape[:2]
            mask = create_circular_mask(h, w, center=(centerX, centerY), radius=2*stdMax)
        
        # remove the nucleus from the mask
        mask[nucWholeMask] = 0
        masks.append(mask)
def save_masks(masks):
    combined_mask = np.empty(())
    for mask in masks:
        combined_mask = np.dstack(combined_mask, mask)
    np.save("masks.npy",combined_mask)

def sample_data():
    global graphData
    # compute
    intensities = np.array([np.sum(samplingImage[mask]) for mask in masks]) / mask_sums
    # initialize minimum intensity
    min_intensity = intensities[0]
    # initialize output list
    temp = [(intensities[0] - min_samplingImage) / (min_intensity - min_samplingImage)]
    for i in range(1, len(intensities)):
        if intensities[i] < min_intensity:
            min_intensity = intensities[i]
        normalized_intensity = (intensities[i] - min_samplingImage) / (min_intensity - min_samplingImage)
        temp.append(normalized_intensity)
    graphData = np.column_stack((graphData, temp))

def display_data():
    global graphData

    graphData = np.delete(graphData, 0, 1)

    fullMask = np.zeros(nucWholeMask.shape)
    for mask in masks:
        fullMask = np.add(fullMask, mask)
    fullMask = fullMask > 0

    # for displaying the main image (subplot must be commented)
    #samplingImage[~fullMask] = 0
    #plt.imshow(samplingImage)

    for i in range(9):
        print(graphData[i])
        plt.subplot(5, 2, i+1).plot(graphData[i])
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The function took {execution_time} seconds to run.")
    plt.show()

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
    multiprocessing.freeze_support()

    # Load the configuration file
    config = []
    with open("config.json") as f:
        config = json.load(f)

    print(config["nuclei_model_location"])

    # for quick running a single image
    #nucDat = np.load(load_path("nucleiMaskLocation.txt"), allow_pickle=True).item()
    #cytoDat = np.load(load_path("cytoMaskLocation.txt"), allow_pickle=True).item()

    dirPath = config["directory_location"]

    imagePaths = os.listdir(dirPath)

    imagePaths = natsorted(imagePaths)
    print(imagePaths)

    samplingImage = plt.imread(os.path.join(dirPath, imagePaths[0]))
    # for quick running a single image
    #samplingImage = plt.imread(load_path("imgLocation.txt"))

    nucModel = models.CellposeModel(gpu=True, pretrained_model=load_path("nucleiModelLocation.txt"))
    cytoModel = models.CellposeModel(gpu=True, pretrained_model=load_path("cytoModelLocation.txt"))

    nucDat = nucModel.eval(samplingImage, channels=[2,0])[0]
    cytoDat = cytoModel.eval(samplingImage, channels=[2,0])[0]

    # plot image with outlines overlaid in red
    #nucOutlines = utils.outlines_list(nucDat['masks'])
    nucOutlines = utils.outlines_list(nucDat)
    #cytoOutlines = utils.outlines_list(cytoDat['masks'])
    cytoOutlines = utils.outlines_list(cytoDat)

    masks = []

    #masks = np.load("masks.npy")

    # for quick running a single image
    #nucWholeMask = nucDat['masks']
    nucWholeMask = nucDat
    nucWholeMask = nucWholeMask > 0

    # for quick running a single image
    #cytoWholeMask = cytoDat['masks']
    cytoWholeMask = cytoDat

    generate_masks()
    #save_masks(masks)

    graphData = np.zeros(len(masks))

    sample_data()

    i = 0
    for imagePath in imagePaths:
        if i > 0:
            samplingImage = plt.imread(os.path.join(dirPath, imagePaths[i]))
            sample_data()
        i+=1

    # stitch together all of the masks
    display_data()
