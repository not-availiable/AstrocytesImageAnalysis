import os
import math
import numpy as np
from cellpose import models, utils
from matplotlib import pyplot as plt
import multiprocessing
import time
import json
from natsort import natsorted
from multiprocessing import Pool, Manager
import concurrent.futures
import AstrocyteAStar as astar
import FullAnalysis as anal
from mpl_point_clicker import clicker
from matplotlib.backend_bases import MouseButton
from PIL import Image, ImageDraw
import cv2

# start timer to measure how long code takes to execute
start_time = time.time()

config = []
pre_dir_path = ""
post_dir_path = ""
pre_image_paths = []
first_sampling_image_path = ""
sampling_image = ""
first_image_sample = ""
first_image_normalized_intensities = []
masks = []
dead_cell = []
close_cell_count = 0
nuclei_centers = []
connection_list = []
min_intensity = 0
max_intensity = 0
pre_offset = 0
post_offset = 0
split_point = 0
stats = []



def load_path(file):
    f = open(file)
    path = f.readline().rstrip()
    f.close()
    return path


def get_center_location(o):
    # takes average
    return o[:, 0].mean(), o[:, 1].mean()


def generate_masks():
    global masks
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
    plt.savefig(os.path.join(config["experiment_name"], "masks_pre.png"), format="png")
    post_image = plt.imread(os.path.join(post_dir_path, post_image_paths[len(post_image_paths)-1]))
    plt.imshow(post_image)
    plt.savefig(os.path.join(config["experiment_name"], "masks_post.png"), format="png")
    plt.clf()
    # im = Image.fromarray(sampling_image)
    # im.save("sampling_image.png")
    # plt.savefig(os.path.join(config["experiment_name"], "sampling_image.png"), format="png")


# def show_and_edit_masks():
#     global masks
#     image = Image.open(first_sampling_image_path)
#     draw = ImageDraw.Draw(image)
#     for o in nuc_outlines:
#         for x, y in o[:,]:
#             draw.point((x, y), fill="red")
#     for c in cyto_outlines:
#         for x, y in c[:,]:
#             draw.point((x, y), fill="blue")
#     while True:
#         image.show()


# def draw_on_image(event, x, y, flags, param):
#     ix = param[0]
#     iy = param[1]
#     drawing = param[2]
#     img = param[3]

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix = x
#         iy = y
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             cv2.rectangle(img, pt1=(ix, iy),
#                           pt2=(x, y),
#                           color=(0, 255, 255),
#                           thickness=-1)
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         cv2.rectangle(img, pt1=(ix, iy),
#                       pt2=(x, y),
#                       color=(0, 255, 255),
#                       thickness=-1)


# def show_and_edit_masks():
#     ix = -1
#     iy = -1
#     drawing = False
#     img = cv2.imread(os.path.join(config["experiment_name"], "sampling_image.png"))
#     param = [ix, iy, drawing, img]
#     cv2.namedWindow(winname="AstrocyteImage")
#     cv2.setMouseCallback("AstrocyteImage", draw_on_image, param)
#     while True:
#         cv2.imshow("AstrocyteImage", img)

#         if cv2.waitKey(10) == 27:
#             break


def save_masks(masks):
    combined_mask = np.empty(())
    for mask in masks:
        combined_mask = np.dstack(combined_mask, mask)
    np.save("masks.npy", combined_mask)


def sample_data(filedata):
    masks = []

    min_intensity = 1
    max_intensity = 1

    filepath, insert_index, masks, first_image_sample, first_image_normalized_intensities = filedata

    sampling_image = plt.imread(filepath)

    temp = []
    # the bg_intensity is the mean of all pixels with a z_score less than 1.21
    bg_intensity = np.mean(sampling_image[np.where((sampling_image - np.mean(sampling_image)) / np.std(sampling_image) < 0.01)])

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

    print("Finished processing frame: " + str(insert_index), flush=True)
    first_image_sample = False
    return temp, insert_index, min_intensity, max_intensity, first_image_sample, first_image_normalized_intensities


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
        if connection_list[i] == 0:
            title_text = "Not Connected"
        elif connection_list[i] == 1:
            title_text = "Networked"
        elif connection_list[i] == 2:
            title_text = "Connected"
        elif connection_list[i] == 3:
            title_text = "Dead Cell"
        plt.title(title_text)
        plt.ylim(min_intensity, max_intensity+.05)
        plt.xlabel("Frame #")
        plt.ylabel("normalized intensity")
        plt.axvline(stats['FWHM_Left_Index'][i], linestyle="dashed")
        plt.axvline(stats['FWHM_Right_Index'][i], linestyle="dashed")
        plt.axhline(stats['Peak_Value'][i], linestyle="dashed")
        plt.savefig(os.path.join(config["experiment_name"], "plot" + str(i) + ".png"), format="png")


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
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

    os.makedirs(os.path.join(os.getcwd(), config["experiment_name"]), exist_ok=True)

    min_intensity = 1
    max_intensity = 1
    min_intensities = []
    max_intensities = []

    dead_cell = 0
    close_cell_count = 0
    nuclei_centers = []

    nuc_model = models.CellposeModel(gpu=True, pretrained_model=str(config["nuclei_model_location"]))
    cyto_model = models.CellposeModel(gpu=True, pretrained_model=str(config["cyto_model_location"]))

    print("Detecting Nuclei", flush=True)
    nuc_dat = nuc_model.eval(sampling_image, channels=[0, 0])[0]
    print("Detecting Cytoplasm", flush=True)
    cyto_dat = cyto_model.eval(sampling_image, channels=[0, 0])[0]

    # plot image with outlines overlaid in red
    nuc_outlines = utils.outlines_list(nuc_dat)
    cyto_outlines = utils.outlines_list(cyto_dat)

    nuc_whole_mask = np.copy(nuc_dat)
    nuc_whole_mask = nuc_whole_mask > 0

    cyto_whole_mask = cyto_dat

    print("Generating Masks", flush=True)
    generate_masks()

    print("Here are the current masks, if there are any mistakes, please fix them")
    # show_and_edit_masks()

    graph_data = np.zeros((len(masks), len(pre_image_paths) + len(post_image_paths)))

    full_image_data = [(first_sampling_image_path, len(pre_image_paths)-1, masks, first_image_sample, first_image_normalized_intensities)]

    # run this image outside of the multiprocessing to gurantee that it happens first
    temp = []
    insert_index = len(pre_image_paths) - 1

    temp, insert_index, min_intensity, max_intensity, first_image_sample, first_image_normalized_intensities = sample_data(full_image_data[0])

    min_intensities.append(min_intensity)
    max_intensities.append(max_intensity)
    graph_data[:, insert_index] = temp

    # add every path but the one that was just sampled
    # then remove that first path so it is not sampled twice
    i = 0
    for image_path in pre_image_paths:
        if i != insert_index:
            full_image_data.append((os.path.join(pre_dir_path, image_path), i, masks, first_image_sample, first_image_normalized_intensities))
        i += 1

    for image_path in post_image_paths:
        full_image_data.append((os.path.join(post_dir_path, image_path), i, masks, first_image_sample, first_image_normalized_intensities))
        i += 1

    full_image_data.pop(0)

    # sample all of the data using multiprocessing
    p = Pool(16)
    for result in p.map(sample_data, full_image_data):
        temp, insert_index, min_intensity, max_intensity, first_image_sample, first_image_normalized_intensities= result
        min_intensities.append(min_intensity)
        max_intensities.append(max_intensity)
        graph_data[:, insert_index] = temp
    p.close()

    min_intensity = np.min(min_intensities)
    max_intensity = np.max(max_intensities)

    split_point = len(pre_image_paths)

    # find the dead cell and the close cells automatically
    # can now run the code overnight because there are no prompts
    max_roi_intensity = 0
    max_roi_intensity_index = 0
    for i, intensities_list in enumerate(graph_data):
        if i == 0:
            max_roi_intensity = np.max(intensities_list)
        else:
            current_max_roi_intensity = np.max(intensities_list)
            if max_roi_intensity < current_max_roi_intensity:
                max_roi_intensity = current_max_roi_intensity
                max_roi_intensity_index = i
            print("max intensity: " + str(max_roi_intensity) + " " + str(current_max_roi_intensity))    
        print("max intensity: " + str(max_roi_intensity))

    dead_cell = max_roi_intensity_index

    print("Dead Cell: " + str(dead_cell), flush=True)

    dead_cell_center = nuclei_centers[dead_cell]
    for i, center in enumerate(nuclei_centers):
        print(math.dist(center, dead_cell_center))
        if math.dist(center, dead_cell_center) < 225 and i != dead_cell:
            close_cell_count += 1

    print("Close Cell Count: " + str(close_cell_count), flush=True)

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
