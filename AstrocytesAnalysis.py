from audioop import mul
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
from tqdm import tqdm
from multiprocessing import Pool

# start timer to measure how long code takes to execute
start_time = time.time()

def load_path(file):
    try:
        f = open(file)
        path = f.readline().rstrip()
        f.close()
        return path
    except Exception as e:
        print(f"Error in load_path: {e}")
        return None

def get_center_location(o):
    try:
        # takes average
        return o[:, 0].mean(), o[:, 1].mean()
    except Exception as e:
        print(f"Error in get_center_location: {e}")
        return None, None

def generate_masks():
    try:
        i = 0
        for o in nucOutlines:
            # nuclei
            # get average x and y
            centerX, centerY = get_center_location(o)
            plt.annotate(str(i), (centerX, centerY), color="white")

            plt.plot(o[:,0], o[:,1], color='r')

            # get standard deviation
            stdX = np.std(o[:,0])
            stdY = np.std(o[:,1])
            stdMax = max(stdX, stdY)

            # cytoplasm
            # see if there is a cytoplasm that is close enough to a nucleus to use
            hasCloseCytoplasm = False
            closeMaskId = 1
            for c in cytoOutlines: 
                cytoCenterX, cytoCenterY = get_center_location(c)
                if math.dist([centerX, centerY], [cytoCenterX, cytoCenterY]) < 50:
                    hasCloseCytoplasm = True
                    break
                closeMaskId+=1

            # use only the relevant part of the cytoplasm mask 
            mask = cytoWholeMask == closeMaskId
            # use original circle method if there are no valid cytoplasm masks
            if not hasCloseCytoplasm:
                plt.plot(centerX, centerY, marker=".", markerfacecolor=(0, 0, 0, 0), markeredgecolor=(0, 0, 1, 1), markersize=2*stdMax)
                h, w = samplingImage.shape[:2]
                mask = create_circular_mask(h, w, center=(centerX, centerY), radius=2*stdMax)
            
            # remove the nucleus from the mask
            mask[nucWholeMask] = 0
            masks.append(mask)
            i += 1
    except Exception as e:
        print(f"Error in generate_masks: {e}")

def save_masks(masks):
    try:
        combined_mask = np.empty(())
        for mask in masks:
            combined_mask = np.dstack(combined_mask, mask)
        np.save("masks.npy",combined_mask)
    except Exception as e:
        print(f"Error in save_masks: {e}")

def sample_data(filedata):
    global first_image_sample
    global first_image_normalized_intensities

    try:
        filepath, insert_index = filedata

        print(insert_index)

        samplingImage = plt.imread(filepath)

        temp = []
        min_intensity = np.min(samplingImage)
        for mask in masks:
            intensity = np.sum(samplingImage[mask]) / np.sum(mask)
            normalized_intensity = (intensity - min_intensity)
            temp.append(normalized_intensity)
            if first_image_sample:
                first_image_normalized_intensities.append(normalized_intensity)
        
        first_image_sample = False
        temp = [i / j for i, j in zip(temp, first_image_normalized_intensities)]

        return temp, insert_index
    except Exception as e:
        print(f"Error occurred while sampling data: {e}")
        return None, None
def display_data(graphData, samplingImage):
    try:
        fullMask = np.zeros(nucWholeMask.shape)
        for mask in masks:
            fullMask = np.add(fullMask, mask)
        fullMask = fullMask > 0

        # Create a writable copy of the samplingImage
        samplingImage_copy = np.copy(samplingImage)

        # for displaying the main image (subplot must be commented)
        samplingImage_copy[~fullMask] = 0
        plt.imshow(samplingImage_copy)

        plt.savefig("masks", format="png")

        split_point = len(pre_image_paths)

        post_offset = list(range(split_point, split_point + len(post_image_paths)))

        for i in range(len(masks)):
            plt.clf()
            # pre graph
            plt.plot(graphData[i][:split_point], color="blue")
            # connecting line
            x_points = np.array([split_point-1, split_point])
            y_points = np.array([graphData[i][split_point-1], graphData[i][split_point]])
            plt.plot(x_points, y_points, color="red")
            
            # Make sure the lengths of post_offset and graphData[i][split_point-1:] are the same
            num_points = min(len(post_offset), len(graphData[i][split_point-1:]))
            plt.plot(post_offset[:num_points], graphData[i][split_point-1:][:num_points], color="red")
            plt.savefig("plot" + str(i), format="png")

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"The function took {execution_time} seconds to run.")

    except ValueError as ve:
        print(f"ValueError: {ve}")
        print("Error occurred while plotting the data.")
    except IndexError as ie:
        print(f"IndexError: {ie}")
        print("Error occurred during indexing. Check the dimensions of arrays.")
    except Exception as e:
        print(f"Error: {e}")
        print("An unexpected error occurred.")
def create_circular_mask(h, w, center=None, radius=None):
    try:
        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask
    except Exception as e:
        print(f"Error in create_circular_mask: {e}")
        return None
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

    # for quick running a single image
    #samplingImage = plt.imread(load_path("imgLocation.txt"))

    nucModel = models.CellposeModel(gpu=True, pretrained_model=str(config["nuclei_model_location"]))
    cytoModel = models.CellposeModel(gpu=True, pretrained_model=str(config["cyto_model_location"]))

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

    graphData = np.zeros((len(masks), len(pre_image_paths) + len(post_image_paths)))
    print(np.shape(graphData))

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
    p.join()
    # stitch together all of the masks
    display_data(graphData, samplingImage)
