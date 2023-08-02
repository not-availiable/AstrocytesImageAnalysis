# Import the necessary libraries
import os
import math
import numpy as np
from cellpose import models, utils
from matplotlib import pyplot as plt
import multiprocessing
import time
import json
from natsort import natsorted
from tqdm import tqdm
from multiprocessing import Pool
from skimage.transform import rescale
import seaborn as sns
from datetime import datetime
import logging
# Set the style of the graphs to whitegrid
sns.set_style("whitegrid")
logging.basicConfig(filename='astrocyte_analysis.log', level=logging.INFO, format='%(asctime)s %(message)s')
# Start a timer to calculate the total execution time
start_time = time.time()

# Function to get the center location of a given object
def get_center_location(o):
    try:
        # Return the mean of the x and y coordinates
        return o[:, 0].mean(), o[:, 1].mean()  
    except Exception as e:
        print(f"Error in get_center_location: {e}")
        return None, None

# Function to generate masks around the identified cells
def generate_masks(): 
    try:
        # Initialize a counter
        i = 0
        # Loop through each nuclei outline
        for o in nucOutlines:
            # Calculate the center of the nuclei
            centerX, centerY = get_center_location(o)
            # Add the center location to the plot
            plt.annotate(str(i), (centerX, centerY), color="white")
            # Plot the outline of the nuclei
            plt.plot(o[:,0], o[:,1], color='r')
            
            # Calculate the standard deviation of the x and y coordinates
            stdX = np.std(o[:,0])
            stdY = np.std(o[:,1])
            stdMax = max(stdX, stdY)

            # Flag to check if there is a cytoplasm close to the nuclei
            hasCloseCytoplasm = False
            closeMaskId = 1
            # Loop through each cytoplasm outline
            for c in cytoOutlines: 
                # Calculate the center of the cytoplasm
                cytoCenterX, cytoCenterY = get_center_location(c)
                # If the distance between the nuclei and cytoplasm is less than 50, set the flag to True
                if math.dist([centerX, centerY], [cytoCenterX, cytoCenterY]) < 50:
                    hasCloseCytoplasm = True
                    break
                closeMaskId += 1

            # Use only the relevant part of the cytoplasm mask
            mask = cytoWholeMask == closeMaskId
            # If there are no valid cytoplasm masks, use a circular mask
            if not hasCloseCytoplasm:
                plt.plot(centerX, centerY, marker=".", markerfacecolor=(0, 0, 0, 0), markeredgecolor=(0, 0, 1, 1), markersize=2*stdMax)
                h, w = samplingImage.shape[:2]
                mask = create_circular_mask(h, w, center=(centerX, centerY), radius=2*stdMax)
            
            # Remove the nucleus from the mask
            mask[nucWholeMask] = 0
            # Add the mask to the list of masks
            masks.append(mask)
            i += 1
    except Exception as e:
        print(f"Error in generate_masks: {e}")

# Function to display the normalized data
def display_normalized_data(graphData, samplingImage, pre_image_paths, post_image_paths):
    try:
        # Initialize an empty mask
        fullMask = np.zeros(samplingImage.shape[:2], dtype=bool)
        # Combine all masks into one
        for mask in masks:
            fullMask = np.logical_or(fullMask, mask)

        # Create a copy of the sampling image with three channels
        samplingImage_copy = np.zeros((samplingImage.shape[0], samplingImage.shape[1], 3), dtype=np.uint8)
        # Copy the green channel from the original sampling image to the copy
        samplingImage_copy[:, :, 1] = samplingImage[:, :, 1]
        # Set the red and blue channels of the masked region to 0
        samplingImage_copy[~fullMask, 0] = 0
        samplingImage_copy[~fullMask, 2] = 0
        # Display the image
        plt.imshow(samplingImage_copy)
        # Save the image as a PNG
        plt.savefig("masks", format="png")

        # Extract timestamps from image names
        pre_timestamps = [extract_timestamp(name) for name in pre_image_paths]
        post_timestamps = [extract_timestamp(name) for name in post_image_paths]

        # Calculate zero time as the midpoint between pre and post images
        zero_time = datetime.fromtimestamp((pre_timestamps[-1].timestamp() + post_timestamps[0].timestamp()) / 2)

        # Convert timestamps to seconds relative to zero time
        pre_timestamps = [(t - zero_time).total_seconds() for t in pre_timestamps]
        post_timestamps = [(t - zero_time).total_seconds() for t in post_timestamps]

        # Loop through each mask
        for i in range(len(masks)):
            fig, ax = plt.subplots()

            # Plot the pre-image data
            ax.plot(pre_timestamps, graphData[i][:len(pre_timestamps)], color="blue", label="Pre-image data")

            # Plot the post-image data
            ax.plot(post_timestamps[:len(graphData[i][len(pre_timestamps):])], graphData[i][len(pre_timestamps):], color="red", label="Post-image data")

            # Plot a connecting line between the last pre-image data point and the first post-image data point
            last_pre_time = pre_timestamps[-1]
            last_pre_intensity = graphData[i][len(pre_timestamps)-1]
            first_post_time = post_timestamps[0]
            first_post_intensity = graphData[i][len(pre_timestamps)]
            ax.plot([last_pre_time, first_post_time], [last_pre_intensity, first_post_intensity], color="green", label="Pre-post connection")

            # Calculate and display the average difference in intensity
            pre_avg = np.mean(graphData[i][:len(pre_timestamps)])
            post_avg = np.mean(graphData[i][len(pre_timestamps):])
            ax.annotate(f'Diff: {post_avg - pre_avg:.2f}', xy=(0.8, 0.75), xycoords='axes fraction')

            # Calculate and display the peak intensity
            peak_intensity_time = pre_timestamps + post_timestamps
            peak_intensity_index = np.argmax(graphData[i])
            peak_intensity = graphData[i][peak_intensity_index]
            ax.annotate(f'Peak: {peak_intensity:.2f}', xy=(peak_intensity_time[peak_intensity_index], peak_intensity),
                        xytext=(0.8, 0.7), xycoords='data', textcoords='axes fraction',
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

            # Add labels and titles
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Normalized Intensity')
            ax.set_title(f'Normalized Intensity vs Time for Mask {i+1}')

            # Set the y-axis limits
            ax.set_ylim([0, 2])
            # Add a grid and legend
            ax.grid(True)
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

            # Adjust layout and save the plot
            fig.tight_layout()
            plt.savefig("plot" + str(i), format="png")

            # Close the figure to free up memory
            plt.close(fig)

        # Calculate and print the execution time
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


# Function to display the raw data
def display_raw_data(rawData, samplingImage, pre_image_paths, post_image_paths):
    try:
        # Initialize an empty mask
        fullMask = np.zeros(samplingImage.shape[:2], dtype=bool)
        # Combine all masks into one
        for mask in masks:
            fullMask = np.logical_or(fullMask, mask)

        # Create a copy of the sampling image with three channels
        samplingImage_copy = np.zeros((samplingImage.shape[0], samplingImage.shape[1], 3), dtype=np.uint8)
        # Copy the green channel from the original sampling image to the copy
        samplingImage_copy[:, :, 1] = samplingImage[:, :, 1]
        # Set the red and blue channels of the masked region to 0
        samplingImage_copy[~fullMask, 0] = 0
        samplingImage_copy[~fullMask, 2] = 0
        # Display the image
        plt.imshow(samplingImage_copy)
        # Save the image as a PNG
        plt.savefig("masks_raw", format="png")

        # Extract timestamps from image names
        pre_timestamps = [extract_timestamp(name) for name in pre_image_paths]
        post_timestamps = [extract_timestamp(name) for name in post_image_paths]

        # Calculate zero time as the midpoint between pre and post images
        zero_time = datetime.fromtimestamp((pre_timestamps[-1].timestamp() + post_timestamps[0].timestamp()) / 2)

        # Convert timestamps to seconds relative to zero time
        pre_timestamps = [(t - zero_time).total_seconds() for t in pre_timestamps]
        post_timestamps = [(t - zero_time).total_seconds() for t in post_timestamps]

        # Loop through each mask
        for i in range(len(masks)):
            fig, ax = plt.subplots()
            # Plot the pre-image data
            ax.plot(pre_timestamps, rawData[i][:len(pre_timestamps)], color="blue", label="Pre-image data")

            # Plot the post-image data
            ax.plot(post_timestamps[:len(rawData[i][len(pre_timestamps):])], rawData[i][len(pre_timestamps):], color="red", label="Post-image data")

            # Plot a connecting line between the last pre-image data point and the first post-image data point
            last_pre_time = pre_timestamps[-1]
            last_pre_intensity_raw = rawData[i][len(pre_timestamps)-1]
            first_post_time = post_timestamps[0]
            first_post_intensity_raw = rawData[i][len(pre_timestamps)]
            ax.plot([last_pre_time, first_post_time], [last_pre_intensity_raw, first_post_intensity_raw], color="green", label="Pre-Post connection")

            # Calculate and display the average difference in intensity
            pre_avg = np.mean(rawData[i][:len(pre_timestamps)])
            post_avg = np.mean(rawData[i][len(pre_timestamps):])
            ax.annotate(f'Diff: {post_avg - pre_avg:.2f}', xy=(0.6, 0.85), xycoords='axes fraction')

            # Calculate and display the peak intensity
            peak_intensity_time = pre_timestamps + post_timestamps
            peak_intensity_index = np.argmax(rawData[i])
            peak_intensity = rawData[i][peak_intensity_index]
            ax.annotate(f'Peak: {peak_intensity:.2f}', xy=(peak_intensity_time[peak_intensity_index], peak_intensity),
                        xytext=(0.6, 0.80), xycoords='data', textcoords='axes fraction',
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

            # Add labels and titles
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Raw Intensity')
            ax.set_title(f'Raw Intensity vs Time for Mask {i+1}')

            # Add a grid and legend
            ax.grid(True)
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

            # Adjust layout and save the plot
            fig.tight_layout()
            plt.savefig("plot_raw" + str(i), format="png")

            # Close the figure to free up memory
            plt.close(fig)

        # Calculate and print the execution time
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

# Function to save the masks to a numpy array file
def save_masks(masks):
    try:
        # Initialize an empty array
        combined_mask = np.empty((masks[0].shape[0], masks[0].shape[1], 0))

        # Loop through each mask and stack it to the combined mask
        for mask in masks:
            if mask.shape == masks[0].shape:  # ensure that all masks have the same size
                combined_mask = np.dstack((combined_mask, mask))
            else:
                print(f"Skipping mask due to size mismatch: expected {masks[0].shape}, got {mask.shape}")
                
        # Save the combined mask to a numpy array file
        np.save("masks.npy", combined_mask)
    except Exception as e:
        print(f"Error in save_masks: {e}")

# Function to sample data from the given file
def sample_data(filedata):
    global first_image_sample
    global first_image_normalized_intensities

    try:
        filepath, insert_index = filedata
        print(insert_index)
        samplingImage = plt.imread(filepath)
        samplingImage = rescale(samplingImage, 0.8, anti_aliasing=True)

        temp = []
        temp_raw = []
        min_intensity = np.min(samplingImage)
        for mask in masks:
            intensity = np.sum(samplingImage[mask]) / np.sum(mask)
            temp_raw.append(intensity)
            normalized_intensity = (intensity - min_intensity)
            temp.append(normalized_intensity)
            if first_image_sample:
                first_image_normalized_intensities.append(normalized_intensity)
        
        first_image_sample = False
        temp = [i / j for i, j in zip(temp, first_image_normalized_intensities)]

        return temp, insert_index, temp_raw
    except Exception as e:
        print(f"Error occurred while sampling data: {e}")
        return None, None, None

# Function to create a circular mask
def create_circular_mask(h, w, center=None, radius=None):
    try:
        if center is None:
            center = (int(w/2), int(h/2))
        if radius is None:
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask
    except Exception as e:
        print(f"Error in create_circular_mask: {e}")
        return None

# Function to extract the timestamp from the given filename
def extract_timestamp(filename):
    from datetime import datetime
    # Extract the timestamp string from the filename
    timestamp_str = filename.split('T')[0] + 'T' + filename.split('T')[1].split('Z')[0]
    # Truncate the last digit from the microseconds
    timestamp_str = timestamp_str[:-1]
    # Correctly parse the timestamp string
    return datetime.strptime(timestamp_str, "%Y-%m-%dT%H_%M_%S_%f")

# Main function
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Load the configuration from a JSON file
    config = []
    with open("config.json") as f:
        config = json.load(f)

    # Reading the directories
    pre_dir_path = config["pre_directory_location"]
    post_dir_path = config["post_directory_location"]
    pre_image_paths = os.listdir(pre_dir_path)
    pre_image_paths = natsorted(pre_image_paths)
    post_image_paths = os.listdir(post_dir_path)
    post_image_paths = natsorted(post_image_paths)

    # Loading the sampling image
    samplingImage = plt.imread(os.path.join(pre_dir_path, pre_image_paths[-1]))
    samplingImage = rescale(samplingImage, 0.8, anti_aliasing=True)

    # Setting the first image sample flag
    first_image_sample = True
    first_image_normalized_intensities = []

    # Creating the cellpose models
    nucModel = models.CellposeModel(gpu=True, pretrained_model=str(config["nuclei_model_location"]))
    cytoModel = models.CellposeModel(gpu=True, pretrained_model=str(config["cyto_model_location"]))

    # Running the cellpose model on the sample image
    start2=time.time()
    nucDat = nucModel.eval(samplingImage, channels=[2,0], progress=tqdm())[0]
    cytoDat = cytoModel.eval(samplingImage, channels=[2,0], progress=tqdm())[0]
    end2=time.time()
    final=end2-start2
    logging.info("Cellpose model evaluation time: %s seconds" % final)

    # Extracting the outlines from the cellpose output
    nucOutlines = utils.outlines_list(nucDat)
    cytoOutlines = utils.outlines_list(cytoDat)

    # Initializing the masks
    masks = []
    nucWholeMask = nucDat
    nucWholeMask = nucWholeMask > 0
    cytoWholeMask = cytoDat

    # Generating the masks
    generate_masks()

    # Initializing the graph data
    graphData = np.zeros((len(masks), len(pre_image_paths) + len(post_image_paths)))
    rawData = np.zeros((len(masks), len(pre_image_paths) + len(post_image_paths)))
    logging.info("Graph data shape: %s" % str(np.shape(graphData)))

    # Setting up the full image data
    full_image_data = [(os.path.join(pre_dir_path, pre_image_paths[-1]), 0)]

    # Sampling the first image
    temp, insert_index, temp_raw = sample_data(full_image_data[0])
    graphData[:,insert_index] = temp
    rawData[:,insert_index] = temp_raw

    # Setting up the full image data for the remaining images
    i = 0
    for image_path in pre_image_paths:
        if i > 0:
            full_image_data.append((os.path.join(pre_dir_path, image_path), i))
        i+=1

    for image_path in post_image_paths:
        full_image_data.append((os.path.join(post_dir_path, image_path), i))
        i+=1

    # Processing the images using multiprocessing
    total_images = len(full_image_data)
    with Pool(4) as p:
        for idx, result in enumerate(p.imap_unordered(sample_data, full_image_data[1:])):
            temp, insert_index, temp_raw = result
            graphData[:, insert_index] = temp
            rawData[:, insert_index] = temp_raw

            # Write progress to file
            with open('progress.txt', 'w') as f:
                f.write(f"{idx+1},{total_images}")

    # Displaying the data
    display_normalized_data(graphData, samplingImage, pre_image_paths, post_image_paths)
    display_raw_data(rawData, samplingImage, pre_image_paths, post_image_paths)

    # Saving the masks
    save_masks(masks)
