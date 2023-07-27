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
from skimage.transform import rescale

# start timer to measure how long code takes to execute
start_time = time.time()

# function to calculate the center location of a mask
def get_center_location(o):
    try:
        # calculates the mean of the coordinates to get the center
        return o[:, 0].mean(), o[:, 1].mean()  
    except Exception as e:
        print(f"Error in get_center_location: {e}")
        return None, None

# function to generate the masks for each cell
def generate_masks():
    try:
        i = 0
        # loop through each nuclei outline
        for o in nucOutlines:
            # calculate center of the nuclei
            centerX, centerY = get_center_location(o)
            # add the center location to the plot
            plt.annotate(str(i), (centerX, centerY), color="white")
            # plot the outline of the nuclei
            plt.plot(o[:,0], o[:,1], color='r')

            # calculate the standard deviation of the x and y coordinates
            stdX = np.std(o[:,0])
            stdY = np.std(o[:,1])
            stdMax = max(stdX, stdY)

            # set flag to check if there is a cytoplasm close to the nuclei
            hasCloseCytoplasm = False
            closeMaskId = 1
            # loop through each cytoplasm outline
            for c in cytoOutlines: 
                # calculate center of the cytoplasm
                cytoCenterX, cytoCenterY = get_center_location(c)
                # if the distance between the nuclei and cytoplasm is less than 50, set the flag to True
                if math.dist([centerX, centerY], [cytoCenterX, cytoCenterY]) < 50:
                    hasCloseCytoplasm = True
                    break
                closeMaskId+=1

            # use only the relevant part of the cytoplasm mask 
            mask = cytoWholeMask == closeMaskId
            # if there are no valid cytoplasm masks, use a circular mask
            if not hasCloseCytoplasm:
                plt.plot(centerX, centerY, marker=".", markerfacecolor=(0, 0, 0, 0), markeredgecolor=(0, 0, 1, 1), markersize=2*stdMax)
                h, w = samplingImage.shape[:2]
                mask = create_circular_mask(h, w, center=(centerX, centerY), radius=2*stdMax)
            
            # remove the nucleus from the mask
            mask[nucWholeMask] = 0
            # add the mask to the list of masks
            masks.append(mask)
            i += 1
    except Exception as e:
        print(f"Error in generate_masks: {e}")

# function to save the masks to a numpy array file
def save_masks(masks):
    try:
        # initialize an empty array
        combined_mask = np.empty(())
        # loop through each mask and stack it to the combined mask
        for mask in masks:
            combined_mask = np.dstack(combined_mask, mask)
        # save the combined mask to a numpy array file
        np.save("masks.npy",combined_mask)
    except Exception as e:
        print(f"Error in save_masks: {e}")

# function to sample the data from the image files
def sample_data(filedata):
    global first_image_sample
    global first_image_normalized_intensities

    try:
        # unpack the filedata into filepath and insert_index
        filepath, insert_index = filedata
        # print the index
        print(insert_index)
        # read the image file
        samplingImage = plt.imread(filepath)
        # downscale the image for faster cellpose readability
        samplingImage = rescale(samplingImage, 0.8, anti_aliasing=True)

        # initialize temp arrays for storing intensity data
        temp = []
        temp_raw = []
        # calculate the minimum intensity of the image
        min_intensity = np.min(samplingImage)
        # loop through each mask
        for mask in masks:
            # calculate the intensity of the mask
            intensity = np.sum(samplingImage[mask]) / np.sum(mask)
            # add the intensity to the raw temp array
            temp_raw.append(intensity)
            # calculate the normalized intensity
            normalized_intensity = (intensity - min_intensity)
            # add the normalized intensity to the temp array
            temp.append(normalized_intensity)
            # if it's the first image, add the normalized intensity to the first image normalized intensities array
            if first_image_sample:
                first_image_normalized_intensities.append(normalized_intensity)
        
        # set the first image sample flag to False after the first image
        first_image_sample = False
        # normalize the temp array
        temp = [i / j for i, j in zip(temp, first_image_normalized_intensities)]

        # return the normalized intensities, index, and raw intensities
        return temp, insert_index, temp_raw
    except Exception as e:
        print(f"Error occurred while sampling data: {e}")
        return None, None, None

# function to display the normalized data
def display_normalized_data(graphData, samplingImage):
    try:
        # initialize an empty mask
        fullMask = np.zeros(samplingImage.shape[:2], dtype=bool)
        # combine all masks into one
        for mask in masks:
            fullMask = np.logical_or(fullMask, mask)

        # create a copy of the sampling image with three channels
        samplingImage_copy = np.zeros((samplingImage.shape[0], samplingImage.shape[1], 3), dtype=np.uint8)
        # copy the green channel from the original sampling image to the copy
        samplingImage_copy[:, :, 1] = samplingImage[:, :, 1]
        # set the red and blue channels of the masked region to 0
        samplingImage_copy[~fullMask, 0] = 0
        samplingImage_copy[~fullMask, 2] = 0
        # display the image
        plt.imshow(samplingImage_copy)
        # save the image as a PNG
        plt.savefig("masks", format="png")

        # calculate the split point between pre and post image paths
        split_point = len(pre_image_paths)
        # calculate the offset for the post image paths
        post_offset = list(range(split_point, split_point + len(post_image_paths)))

        # loop through each mask
        for i in range(len(masks)):
            # clear the current figure
            plt.clf()
            # plot the pre image data
            plt.plot(graphData[i][:split_point], color="blue")
            # plot the connecting line
            x_points = np.array([split_point-1, split_point])
            y_points = np.array([graphData[i][split_point-1], graphData[i][split_point]])
            plt.plot(x_points, y_points, color="red")
            # plot the post image data
            num_points = min(len(post_offset), len(graphData[i][split_point-1:]))
            plt.plot(post_offset[:num_points], graphData[i][split_point-1:][:num_points], color="red")
            # save the plot as a PNG
            plt.savefig("plot" + str(i), format="png")

        # print the execution time
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

# function to display the raw data
def display_raw_data(rawData, samplingImage):
    try:
        # initialize an empty mask
        fullMask = np.zeros(samplingImage.shape[:2], dtype=bool)
        # combine all masks into one
        for mask in masks:
            fullMask = np.logical_or(fullMask, mask)

        # create a copy of the sampling image with three channels
        samplingImage_copy = np.zeros((samplingImage.shape[0], samplingImage.shape[1], 3), dtype=np.uint8)
        # copy the green channel from the original sampling image to the copy
        samplingImage_copy[:, :, 1] = samplingImage[:, :, 1]
        # set the red and blue channels of the masked region to 0
        samplingImage_copy[~fullMask, 0] = 0
        samplingImage_copy[~fullMask, 2] = 0
        # display the image
        plt.imshow(samplingImage_copy)
        # save the image as a PNG
        plt.savefig("masks_raw", format="png")

        # calculate the split point between pre and post image paths
        split_point = len(pre_image_paths)
        # calculate the offset for the post image paths
        post_offset = list(range(split_point, split_point + len(post_image_paths)))

        # loop through each mask
        for i in range(len(masks)):
            # clear the current figure
            plt.clf()
            # plot the pre image data
            plt.plot(rawData[i][:split_point], color="blue")
            # plot the connecting line
            x_points = np.array([split_point-1, split_point])
            y_points = np.array([rawData[i][split_point-1], rawData[i][split_point]])
            plt.plot(x_points, y_points, color="red")
            # plot the post image data
            num_points = min(len(post_offset), len(rawData[i][split_point-1:]))
            plt.plot(post_offset[:num_points], rawData[i][split_point-1:][:num_points], color="red")
            # save the plot as a PNG
            plt.savefig("plot_raw" + str(i), format="png")

        # print the execution time
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

# function to create a circular mask
def create_circular_mask(h, w, center=None, radius=None):
    try:
        # if no center is provided, use the center of the image
        if center is None:
            center = (int(w/2), int(h/2))
        # if no radius is provided, use the smallest distance between the center and the image edges
        if radius is None:
            radius = min(center[0], center[1], w-center[0], h-center[1])

        # generate a grid of coordinates
        Y, X = np.ogrid[:h, :w]
        # calculate the distance from each coordinate to the center
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        # generate a mask for coordinates within the radius
        mask = dist_from_center <= radius
        return mask
    except Exception as e:
        print(f"Error in create_circular_mask: {e}")
        return None

# main script
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # load the config file
    config = []
    with open("config.json") as f:
        config = json.load(f)

    # get the directory paths for pre and post images
    pre_dir_path = config["pre_directory_location"]
    post_dir_path = config["post_directory_location"]
    # get the image paths within each directory
    pre_image_paths = os.listdir(pre_dir_path)
    pre_image_paths = natsorted(pre_image_paths)
    post_image_paths = os.listdir(post_dir_path)
    post_image_paths = natsorted(post_image_paths)

    # read the first pre image
    samplingImage = plt.imread(os.path.join(pre_dir_path, pre_image_paths[0]))
    # downscale the image for faster cellpose readability
    samplingImage = rescale(samplingImage, 0.8, anti_aliasing=True)

    # set the first image sample flag to True
    first_image_sample = True
    # initialize the first image normalized intensities array
    first_image_normalized_intensities = []

    # load the Cellpose models for nuclei and cytoplasm
    nucModel = models.CellposeModel(gpu=True, pretrained_model=str(config["nuclei_model_location"]))
    cytoModel = models.CellposeModel(gpu=True, pretrained_model=str(config["cyto_model_location"]))

    # start the timer for cellpose
    start2=time.time()
    # evaluate the image with the Cellpose models
    nucDat = nucModel.eval(samplingImage, channels=[2,0], progress=tqdm())[0]
    cytoDat = cytoModel.eval(samplingImage, channels=[2,0], progress=tqdm())[0]
    # stop the timer for cellpose
    end2=time.time()
    # print the time taken for cellpose
    final=end2-start2
    print(final)

    # get the outlines for each nucleus and cytoplasm
    nucOutlines = utils.outlines_list(nucDat)
    cytoOutlines = utils.outlines_list(cytoDat)

    # initialize the masks array
    masks = []
    # get the whole mask for each nucleus
    nucWholeMask = nucDat
    # binarize the nucleus mask
    nucWholeMask = nucWholeMask > 0
    # get the whole mask for each cytoplasm
    cytoWholeMask = cytoDat

    # generate the masks
    generate_masks()

    # initialize the graph data and raw data arrays
    graphData = np.zeros((len(masks), len(pre_image_paths) + len(post_image_paths)))
    rawData = np.zeros((len(masks), len(pre_image_paths) + len(post_image_paths)))
    print(np.shape(graphData))

    # initialize the full image data array with the first pre image
    full_image_data = [(os.path.join(pre_dir_path, pre_image_paths[0]), 0)]
    # sample the data from the first pre image
    temp, insert_index, temp_raw = sample_data(full_image_data[0])
    # add the data to the graph data and raw data arrays
    graphData[:,insert_index] = temp
    rawData[:,insert_index] = temp_raw

    # add the rest of the pre images to the full image data array
    i = 0
    for image_path in pre_image_paths:
        if i > 0:
            full_image_data.append((os.path.join(pre_dir_path, image_path), i))
        i+=1

    # add the post images to the full image data array
    for image_path in post_image_paths:
        full_image_data.append((os.path.join(post_dir_path, image_path), i))
        i+=1

    # create a multiprocessing pool
    p = Pool(16)
    # map the sample_data function to the full image data array
    for result in p.map(sample_data, full_image_data):
        # unpack the result into temp, insert_index, and temp_raw
        temp, insert_index, temp_raw = result
        # add the data to the graph data and raw data arrays
        graphData[:,insert_index] = temp
        rawData[:,insert_index] = temp_raw

    # close the multiprocessing pool
    p.close()
    p.join()

    # save the raw data to a numpy array file
    np.save("raw_data.npy", rawData)

    # display the normalized data
    display_normalized_data(graphData, samplingImage)
    # display the raw data
    display_raw_data(rawData, samplingImage)
