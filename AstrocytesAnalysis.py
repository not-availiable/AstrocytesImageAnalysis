import os
import math
import numpy as np
from cellpose import models, utils
from matplotlib import pyplot as plt
import multiprocessing
import time
import json
from natsort import natsorted
from multiprocessing import Pool
import AstrocyteAStar as astar
import FullAnalysis as anal

# start timer to measure how long code takes to execute
start_time = time.time()


def load_path(file):
    f = open(file)
    path = f.readline().rstrip()
    f.close()
    return path


def get_center_location(o):
    # takes average
    return o[:, 0].mean(), o[:, 1].mean()


def generate_masks():
    global dead_cell
    global close_cell_count
    global nuclei_centers

    i = 0
    for o in nuc_outlines:
        # nuclei
        # get average x and y
        center_x, center_y = get_center_location(o)
        nuclei_centers.append((center_x, center_y))
        plt.annotate(str(i), (center_x, center_y), color="white")

        plt.plot(o[:, 0], o[:, 1], color='r')

        # get standard deviation
        std_x = np.std(o[:, 0])
        std_y = np.std(o[:, 1])
        std_max = max(std_x, std_y)

        # cytoplasm
        # see if there is a cytoplasm that is close enough to a nucleus to use
        has_close_cytoplasm = False
        close_mask_id = 1
        for c in cyto_outlines:
            if i == 0:
                plt.plot(c[:, 0], c[:, 1], color='r')
            cyto_center_x, cyto_center_y = get_center_location(c)
            if math.dist([center_x, center_y], [cyto_center_x, cyto_center_y]) < 50:
                has_close_cytoplasm = True
                break
            close_mask_id += 1

        # use only the relavant part of the cytoplasm mask
        mask = cyto_whole_mask == close_mask_id
        # use original circle method if there are no valid cytoplasm masks
        if not has_close_cytoplasm:
            h, w = sampling_image.shape[:2]
            mask = create_circular_mask(h, w, center=(center_x, center_y), radius=2*std_max)

        # remove the nucleus from the mask
        mask[nuc_whole_mask] = 0
        masks.append(mask)
        i += 1

    plt.imshow(sampling_image)
    plt.savefig("masks_pre.png", format="png")
    post_image = plt.imread(os.path.join(post_dir_path, post_image_paths[len(post_image_paths)-1]))
    plt.imshow(post_image)
    plt.savefig("masks_post.png", format="png")


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
    combined_mask = np.empty(())
    for mask in masks:
        combined_mask = np.dstack(combined_mask, mask)
    np.save("masks.npy", combined_mask)


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
    # global graphData
    global first_image_sample
    global first_image_normalized_intensities

    min_intensity = 1
    max_intensity = 1

    filepath, insert_index = filedata

    sampling_image = plt.imread(filepath)

    temp = []
    # the bg_intensity is the mean of all pixels with a z_score less than 1.21
    bg_intensity = np.mean(sampling_image[np.where((sampling_image - np.mean(sampling_image)) / np.std(sampling_image) < 1.21)])

    for mask in masks:
        intensity = np.sum(sampling_image[mask]) / np.sum(mask)
        normalized_intensity = (intensity - bg_intensity)
        temp.append(normalized_intensity)
        if first_image_sample:
            first_image_normalized_intensities.append(normalized_intensity)

    # divide intensities by the original intensities
    temp = [i / j for i, j in zip(temp, first_image_normalized_intensities)]

    for value in temp:
        if value < min_intensity:
            min_intensity = value
        if value > max_intensity:
            max_intensity = value

    print("Finished processing frame: " + str(insert_index))
    first_image_sample = False
    return temp, insert_index, min_intensity, max_intensity


def display_data(graph_data):
    global connection_list
    global min_intensity
    global max_intensity
    global pre_offset
    global post_offset
    global split_point
    global stats

    graph_data = np.delete(graph_data, len(graph_data), 1)

    for i in range(len(masks)):
        plt.clf()
        # pre graph
        plt.plot(pre_offset, graph_data[i][:split_point], color="blue")
        # post graph
        plt.plot(post_offset, graph_data[i][split_point-1:], color="red")
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
        plt.ylim(min_intensity, max_intensity+.05)
        plt.xlabel("Frame #")
        plt.ylabel("normalized intensity")
        plt.axvline(stats['FWHM_Left_Index'][i], linestyle="dashed")
        plt.axvline(stats['FWHM_Right_Index'][i], linestyle="dashed")
        plt.axhline(stats['Peak_Value'][i], linestyle="dashed")
        plt.savefig("plot" + str(i) + ".png", format="png")


# Function to create a circular mask
def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
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

    # Load the configuration from a JSON file
    config = []
    with open("config.json") as f:
        config = json.load(f)

    pre_dir_path = config["pre_directory_location"]
    post_dir_path = config["post_directory_location"]

    # get all of the filepaths from the pre and post image folders
    # ignoring hidden folders to fix DS_Store issue
    # natsorted is necessary because the files are not sorted in the correct order by default
    pre_image_paths = os.listdir(pre_dir_path)
    valid_paths = []
    for i, path in enumerate(pre_image_paths):
        if not path.startswith("."):
            valid_paths.append(pre_image_paths[i])
    pre_image_paths = valid_paths
    pre_image_paths = natsorted(pre_image_paths)
    post_image_paths = os.listdir(post_dir_path)
    valid_paths = []
    for i, path in enumerate(post_image_paths):
        if not path.startswith("."):
            valid_paths.append(post_image_paths[i])
    post_image_paths = valid_paths
    post_image_paths = natsorted(post_image_paths)

    # variable definitions
    # first sampling image is the image that every other image will be normalized in relation to
    first_sampling_image_path = os.path.join(pre_dir_path, pre_image_paths[len(pre_image_paths)-1])
    sampling_image = plt.imread(first_sampling_image_path)
    first_image_sample = True
    first_image_normalized_intensities = []

    min_intensity = 1
    max_intensity = 1
    min_intensities = []
    max_intensities = []

    dead_cell = 0
    close_cell_count = 0
    nuclei_centers = []

    nuc_model = models.CellposeModel(gpu=True, pretrained_model=str(config["nuclei_model_location"]))
    cyto_model = models.CellposeModel(gpu=True, pretrained_model=str(config["cyto_model_location"]))

    print("Detecting Nuclei")
    nuc_dat = nuc_model.eval(sampling_image, channels=[2, 0])[0]
    print("Detecting Cytoplasm")
    cyto_dat = cyto_model.eval(sampling_image, channels=[2, 0])[0]

    # plot image with outlines overlaid in red
    nuc_outlines = utils.outlines_list(nuc_dat)
    cyto_outlines = utils.outlines_list(cyto_dat)

    # Initializing the masks
    masks = []

    nuc_whole_mask = nuc_dat
    nuc_whole_mask = nuc_whole_mask > 0

    cyto_whole_mask = cyto_dat

    print("Generating Masks")
    generate_masks()

    graph_data = np.zeros((len(masks), len(pre_image_paths) + len(post_image_paths)))

    full_image_data = [(first_sampling_image_path, len(pre_image_paths)-1)]

    # run this image outside of the multiprocessing to gurantee that it happens first
    temp = []
    insert_index = len(pre_image_paths) - 1

    temp, insert_index, min_intensity, max_intensity = sample_data(full_image_data[0])
    min_intensities.append(min_intensity)
    max_intensities.append(max_intensity)
    graph_data[:, insert_index] = temp

    # add every path but the one that was just sampled
    # then remove that first path so it is not sampled twice
    i = 0
    for image_path in pre_image_paths:
        if i != insert_index:
            full_image_data.append((os.path.join(pre_dir_path, image_path), i))
        i += 1

    for image_path in post_image_paths:
        full_image_data.append((os.path.join(post_dir_path, image_path), i))
        i += 1

    full_image_data.pop(0)

    # sample all of the data using multiprocessing
    p = Pool(16)
    for result in p.map(sample_data, full_image_data):
        temp, insert_index, min_intensity, max_intensity = result
        min_intensities.append(min_intensity)
        max_intensities.append(max_intensity)
        graph_data[:, insert_index] = temp
    p.close()

    min_intensity = np.min(min_intensities)
    max_intensity = np.max(max_intensities)

    split_point = len(pre_image_paths)

    # find the dead cell and the close cells automatically
    # can now run the code overnight because there are no prompts
    min_roi_intensity = 0
    min_roi_intensity_index = 0
    for i, intensities_list in enumerate(graph_data):
        if i == 0:
            min_roi_intensity = np.min(intensities_list)
        else:
            current_min_roi_intensity = np.min(intensities_list)
            if min_roi_intensity > current_min_roi_intensity:
                min_roi_intensity = current_min_roi_intensity
                min_roi_intensity_index = i

    dead_cell = min_roi_intensity_index

    print("Dead Cell: " + str(dead_cell))

    dead_cell_center = nuclei_centers[dead_cell]
    for i, center in enumerate(nuclei_centers):
        print(math.dist(center, dead_cell_center))
        if math.dist(center, dead_cell_center) < 225 and i != dead_cell:
            close_cell_count += 1

    print("Close Cell Count: " + str(close_cell_count))

    # get connections from astar algorithm
    connection_list = astar.run_astar_algorithm(first_sampling_image_path,
                                                nuc_dat, dead_cell,
                                                close_cell_count)
    # offset the pre and post image graphs so they line up
    pre_offset = []
    for i in range(0, len(pre_image_paths)):
        pre_offset.append(i)

    post_offset = []
    for i in range(len(pre_image_paths) - 1, len(pre_image_paths) + len(post_image_paths) - 1):
        post_offset.append(i)

    # create the csv and pass fwhm and max to display_data
    stats = anal.get_stats(nuc_dat, len(masks), pre_offset + post_offset, graph_data, dead_cell)

    # create all of the graphs
    display_data(graph_data)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The function took {execution_time} seconds to run.")
