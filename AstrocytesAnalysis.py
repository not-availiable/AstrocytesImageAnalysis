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
from tqdm import tqdm
from multiprocessing import Pool
import AstrocyteAStar as astar

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
    global dead_cell
    global close_cell_count
    
    i = 0
    for o in nucOutlines:
        # nuclei
        # get average x and y
        centerX, centerY = get_center_location(o)
        plt.annotate(str(i), (centerX, centerY), color="white")

        plt.plot(o[:,0], o[:,1], color='r')

        #get standard deviation
        stdX = np.std(o[:,0])
        stdY = np.std(o[:,1])
        stdMax = max(stdX, stdY)

        # cytoplasm
        # see if there is a cytoplasm that is close enough to a nucleus to use
        hasCloseCytoplasm = False
        closeMaskId = 1
        for c in cytoOutlines: 
            if i == 0:
                plt.plot(c[:,0], c[:,1], color='r') 
            cytoCenterX, cytoCenterY = get_center_location(c)
            if math.dist([centerX, centerY], [cytoCenterX, cytoCenterY]) < 50:
                hasCloseCytoplasm = True
                break
            closeMaskId+=1

        # use only the relavant part of the cytoplasm mask 
        mask = cytoWholeMask == closeMaskId
        # use original circle method if there are no valid cytoplasm masks
        if not hasCloseCytoplasm:
            #plt.plot(centerX, centerY, marker=".", markerfacecolor=(0, 0, 0, 0), markeredgecolor=(0, 0, 1, 1), markersize=2*stdMax)
            h, w = samplingImage.shape[:2]
            mask = create_circular_mask(h, w, center=(centerX, centerY), radius=2*stdMax)
        
        # remove the nucleus from the mask
        mask[nucWholeMask] = 0
        masks.append(mask)
        i+=1
    
    # fullMask = np.zeros(nucWholeMask.shape)
    # for mask in masks:
    #     fullMask = np.add(fullMask, mask)
    # fullMask = fullMask > 0
    
    # samplingImage[~fullMask] = 0
    plt.imshow(samplingImage)
    plt.savefig("masks_pre", format="png")
    post_image = plt.imread(os.path.join(post_dir_path, post_image_paths[len(post_image_paths)-1]))
    plt.imshow(post_image)
    plt.savefig("masks_post", format="png")
    print("please open and compare the two mask images that were just generated in the folder and type the number of the dead cell")
    dead_cell = input()
    dead_cell = int(dead_cell)
    print("please type the number of cells that are close to the dead cell")
    close_cell_count = input()
    close_cell_count = int(close_cell_count)

def save_masks(masks):
    combined_mask = np.empty(())
    for mask in masks:
        combined_mask = np.dstack(combined_mask, mask)
    np.save("masks.npy",combined_mask)

def sample_data(filedata):
    #global graphData
    global first_image_sample
    global first_image_normalized_intensities

    filepath, insert_index = filedata

    samplingImage = plt.imread(filepath)

    temp = []
    min_intensity = np.min(samplingImage)
    print("Minimum Intensity: " + str(min_intensity))
    for mask in masks:
        intensity = np.sum(samplingImage[mask]) / np.sum(mask)
        normalized_intensity = (intensity - min_intensity)
        temp.append(normalized_intensity)
        if first_image_sample:
            first_image_normalized_intensities.append(normalized_intensity)
    
    first_image_sample = False
    temp = [i / j for i, j in zip(temp, first_image_normalized_intensities)]

    #graphData[:,insert_index] = temp
    print("Finished processing frame: " + str(insert_index))
    return temp, insert_index

def display_data(graphData):
    global connection_list
    graphData = np.delete(graphData, len(graphData), 1)

    split_point = len(pre_image_paths)

    pre_offset = []
    for i in range(0, len(pre_image_paths)):
        pre_offset.append(i*3)

    post_offset = []
    for i in range(len(pre_image_paths) - 1, len(pre_image_paths) + len(post_image_paths) - 1):
        post_offset.append(i*3)

    for i in range(len(masks)):
        plt.clf()
        # pre graph
        plt.plot(pre_offset, graphData[i][:split_point], color="blue")
        # connecting line
        # x_points = np.array([(split_point-1) * 3, split_point * 3])
        # y_points = np.array([graphData[i][split_point-1], graphData[i][split_point]])
        # plt.plot(x_points, y_points, color="green")
        # post graph
        plt.plot(post_offset, graphData[i][split_point-1:], color="red")
        title_text = ""
        match connection_list[i]:
            case 0:
                title_text = "Not Connected"
            case 1:
                title_text = "Networked"
            case 2:
                title_text = "Connected"
            case 3:
                title_text = "Dead Cell"
        plt.title(title_text)
        plt.xlabel("time (seconds)")
        plt.ylabel("normalized intensity")
        plt.savefig("plot" + str(i), format="png")

    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The function took {execution_time} seconds to run.")

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

    # for quick running a single image
    #nucDat = np.load(load_path("nucleiMaskLocation.txt"), allow_pickle=True).item()
    #cytoDat = np.load(load_path("cytoMaskLocation.txt"), allow_pickle=True).item()

    pre_dir_path = config["pre_directory_location"]
    post_dir_path = config["post_directory_location"]

    pre_image_paths = os.listdir(pre_dir_path)
    pre_image_paths = natsorted(pre_image_paths)

    post_image_paths = os.listdir(post_dir_path)
    post_image_paths = natsorted(post_image_paths)

    samplingImage = plt.imread(os.path.join(pre_dir_path, pre_image_paths[0]))
    first_image_sample = True
    first_image_normalized_intensities = []
    
    dead_cell = 0
    close_cell_count = 0

    # for quick running a single image
    #samplingImage = plt.imread(load_path("imgLocation.txt"))

    nucModel = models.CellposeModel(gpu=True, pretrained_model=str(config["nuclei_model_location"]))
    cytoModel = models.CellposeModel(gpu=True, pretrained_model=str(config["cyto_model_location"]))

    print("Detecting Nuclei")
    nucDat = nucModel.eval(samplingImage, channels=[2,0])[0]
    print("Detecting Cytoplasm")
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

    print("Generating Masks")
    generate_masks()

    connection_list = astar.runAStarAlgorithm(os.path.join(pre_dir_path, pre_image_paths[0]), nucDat, 576, dead_cell, close_cell_count)

    graphData = np.zeros((len(masks), len(pre_image_paths) + len(post_image_paths)))

    full_image_data = [(os.path.join(pre_dir_path, pre_image_paths[0]), 0)]

    temp = []
    insert_index = 0

    temp, insert_index = sample_data(full_image_data[0])
    graphData[:,insert_index] = temp

    i = 0
    for image_path in pre_image_paths:
        if i > 0:
            full_image_data.append((os.path.join(pre_dir_path, image_path), i))
        i+=1

    for image_path in post_image_paths:
        full_image_data.append((os.path.join(post_dir_path, image_path), i))
        i+=1

    p = Pool(16)
    for result in p.map(sample_data, full_image_data):
        temp, insert_index = result
        graphData[:,insert_index] = temp

    p.close()

    # stitch together all of the masks
    display_data(graphData)
