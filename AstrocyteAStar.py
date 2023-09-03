# Imports for the code
import numpy as np
import heapq
import matplotlib.pyplot as plt
import cv2
import math
from scipy import ndimage
from cellpose import models, utils

# GLOBAL VARIABLES

SIZE = 576

# Test of A* Algorithm using a premade array


def is_valid(row, col):
    return 0 <= row < SIZE and 0 <= col < SIZE


def is_unblocked(grid, row, col):
    return grid[row][col] == 1


def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]


def calculate_h_value(row, col, dest):
    return math.sqrt((row - dest[0]) ** 2 + (col - dest[1]) ** 2)


def trace_path(cell_details, dest):
    row, col = dest
    path = []
    while not (cell_details[row][col]['parent_i'] == row and cell_details[row][col]['parent_j'] == col):
        path.append((row, col))
        temp_row, temp_col = cell_details[row][col]['parent_i'], cell_details[row][col]['parent_j']
        row, col = temp_row, temp_col
    path.append((row, col))
    return path


def a_star_search(grid, src, dest, si, di):
    if not is_valid(src[0], src[1]):
        print("Source is invalid")
        return

    if not is_valid(dest[0], dest[1]):
        print("Destination is invalid")
        return

    if not is_unblocked(grid, src[0], src[1]) or not is_unblocked(grid, dest[0], dest[1]):
        print("Source or the destination is blocked")
        return

    if is_destination(src[0], src[1], dest):
        print("We are already at the destination")
        return

    closed_list = [[False for _ in range(SIZE)] for _ in range(SIZE)]
    cell_details = [
        [{"parent_i": -1, "parent_j": -1, "f": math.inf, "g": math.inf, "h": math.inf} for _ in range(SIZE)]
        for _ in range(SIZE)
    ]

    i, j = src
    cell_details[i][j]["f"] = 0.0
    cell_details[i][j]["g"] = 0.0
    cell_details[i][j]["h"] = 0.0
    cell_details[i][j]["parent_i"] = i
    cell_details[i][j]["parent_j"] = j

    open_list = [(0.0, (i, j))]

    while open_list:
        f, (i, j) = heapq.heappop(open_list)
        closed_list[i][j] = True
        for new_i, new_j in [(i - 1, j), (i + 1, j), (i, j + 1), (i, j - 1), (i - 1, j + 1), (i - 1, j - 1),(i + 1, j + 1), (i + 1, j - 1)]:
            if is_valid(new_i, new_j):
                if is_destination(new_i, new_j, dest):
                    cell_details[new_i][new_j]["parent_i"] = i
                    cell_details[new_i][new_j]["parent_j"] = j
                    print("The destination cell is found")
                    connection_map[si] = 2
                    return trace_path(cell_details, dest)

                if not closed_list[new_i][new_j] and is_unblocked(grid, new_i, new_j):
                    g_new = cell_details[i][j]["g"] + 1.0 if i == new_i or j == new_j else cell_details[i][j]["g"] + 1.414
                    h_new = calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new
                    if cell_details[new_i][new_j]["f"] == math.inf or cell_details[new_i][new_j]["f"] > f_new:
                        heapq.heappush(open_list, (f_new, (new_i, new_j)))
                        cell_details[new_i][new_j]["f"] = f_new
                        cell_details[new_i][new_j]["g"] = g_new
                        cell_details[new_i][new_j]["h"] = h_new
                        cell_details[new_i][new_j]["parent_i"] = i
                        cell_details[new_i][new_j]["parent_j"] = j

    print("Failed to find a direct connection")
    print("Finding closest connection")
    # Goal is not reachable, find the nearest reachable square
    min_distance = float('inf')
    nearest_square = None
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if closed_list[x][y]:
                distance = calculate_h_value(x, y, dest)
                if distance < min_distance:
                    min_distance = distance
                    nearest_square = (x, y)
    connection_map[si] = 1
    return trace_path(cell_details, nearest_square)

# Example usage:

# grid = [
#     [1, 0, 1, 0, 1],
#     [1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 0],
#     [1, 0, 1, 0, 0],
#     [1, 0, 1, 0, 1]
# ]

# start = (0, 0)
# goal = (4, 4)
# path = a_star_search(grid, start, goal)
# for p in path:
#     print("->", p, end=" ")
# print()


# Getting a test image (not being used in the model yet)
def run_astar_algorithm(file_path_name_to_tiff, nuc_dat, shockwaved_cell, close_cell_count):
    image = cv2.imread(file_path_name_to_tiff)
    img = np.copy(image)
    img[(img - np.mean(img)) / np.std(img) < 1.21] = 0
    img[:, :, 0] = 0
    img[:, :, 2] = 0

    nuc_outlines = utils.outlines_list(nuc_dat)
    nuc_whole_mask = nuc_dat
    nuc_whole_mask = nuc_whole_mask > 0

    img[nuc_whole_mask, 1] = image[nuc_whole_mask, 1]
    # plt.imshow(img)
    # plt.title("Removed Background")
    # plt.show()
    img = cv2.resize(img, (SIZE, SIZE))
    # plt.imshow(img)
    # plt.title("Shrunken Image")
    # plt.show()
    centers = []
    print("Removed Background")
    # plt.imshow(img)
    for outline in nuc_outlines:
        centers.append((int(outline[:, 1].mean()) // 4, int(outline[:, 0].mean())//4))
    # plt.plot(outline[:,0]//4, outline[:,1]//4, color='r')
    # plt.show()
    print("Got Centers")
    grid = np.copy(img)[:, :, 1]
    print("Created Grid")
    global connection_map
    connection_map = np.zeros((len(centers)))  # 0 = not connected, 1 = networked, 2 = connected
    paths = []
    connection_map[shockwaved_cell] = 3
    print("Created Map")
    for i in range(len(centers)):
        if i == shockwaved_cell:
            paths.append([0])
            continue
        grid[grid > 0] = 1
        start = centers[i]
        goal = centers[shockwaved_cell]
        print(f"Got Centers For Cell {i}")
        path = a_star_search(grid, start, goal, i, shockwaved_cell)
        paths.append(path)
        print(f"Got Path For Cell {i}")
        for point in path:
            x = point[0]
            y = point[1]
            grid[x][y] = 255
        full_distance = math.sqrt((abs(start[0]-goal[0]))**2 + (abs(start[1]-goal[1]))**2)
        if (connection_map[i] == 1):
            connection_map[i] = int(1.25 > len(path)/full_distance and len(path)/full_distance > .75)
            # print(len(path))
        print(f"Finished Cell {i}")
        img[:, :, 0] = np.copy(grid[:, :])
        img[start[0], start[1], 2] = 255
        img[goal[0], goal[1], 2] = 255
        # plt.imshow(img)
        # plt.title(f"Cell {i} to Cell {shockwavedCell}")
        # plt.show()
    paths = np.array(paths, dtype=object)
    dist_arr = []
    for arr in paths[np.where(connection_map == 2)]:
        dist_arr.append(len(arr))
    dist_arr = np.sort(dist_arr)
    min_distance = dist_arr[close_cell_count]
    max_distance = max(len(arr) for arr in paths[np.where(connection_map == 2)])
    for i in range(len(centers)):
        if (connection_map[i] == 2):
            connection_map[i] = int(len(paths[i]) < ((min_distance + max_distance)//(((len(centers))//4) + 1) )) + 1
            # print(len(paths[i]))
    return connection_map

# EXAMPLE:

# image = cv2.imread('/Users/connor/Downloads/TrainingSet/32_0.tiff')
# nucModel = models.CellposeModel(gpu=True, pretrained_model=str('/Users/connor/Downloads/TrainingSet/models/AstroNuclei2'))
# nucDat = nucModel.eval(image, channels=[2,0])[0]
# runAStarAlgorithm('/Users/connor/Downloads/TrainingSet/32_0.tiff', nucDat, 6, 6)