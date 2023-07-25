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
                    connectionMap[si] = 2
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

    print("Failed to find the Destination Cell")
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
    connectionMap[si] = 1
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
def runAStarAlgorithm(filePathNameToTiff, filePathNameToNucleiModel, size):
    SIZE = size
    image = cv2.imread(filePathNameToTiff)
    img = np.copy(image)
    background = int(abs(np.mean(img) - np.median(img))*3)
    print(background)
    img[img < background] = 0
    img[:,:,0] = 0
    img[:,:,2] = 0
    
    nucModel = models.CellposeModel(gpu=True, pretrained_model=str(filePathNameToNucleiModel))
    nucDat = nucModel.eval(img, channels=[2,0])[0]
    nucOutlines = utils.outlines_list(nucDat)
    nucWholeMask = nucDat
    nucWholeMask = nucWholeMask > 0
    
    img[nucWholeMask==True, 1] = 255
    plt.imshow(img)
    plt.title("Removed Background")
    plt.show()
    img = cv2.resize(img, (SIZE, SIZE))
    plt.imshow(img)
    plt.title("Shrunken Image")
    plt.show()
    centers = []
    print("Removed Background")
    plt.imshow(img)
    for outline in nucOutlines:
        centers.append(( int(outline[:, 1].mean())//4, int(outline[:, 0].mean())//4 ))
        plt.plot(outline[:,0]//4, outline[:,1]//4, color='r')
    plt.show()
    print("Got Centers")
    grid = np.copy(img)[:,:,1]
    print("Created Grid")
    global connectionMap
    connectionMap = np.zeros((len(centers))) # 0 = not connected, 1 = networked, 2 = connected
    paths = []
    shockwavedCell = 4
    connectionMap[shockwavedCell] = 3
    print("Created Map")
    for i in range(len(centers)):
        if i == shockwavedCell:
            paths.append([0])
            continue
        grid[grid > 0] = 1
        start = centers[i]
        goal = centers[shockwavedCell]
        print(f"Got Centers For Cell {i}")
        path = a_star_search(grid, start, goal, i, shockwavedCell)
        paths.append(path)
        print(f"Got Path For Cell {i}")
        for point in path:
            x = point[0]
            y = point[1]
            grid[x][y] = 255
        fullDistance = math.sqrt( (abs(start[0]-goal[0]))**2 + (abs(start[1]-goal[1]))**2 )
        if (connectionMap[i] == 1):
            connectionMap[i] = int(1.25 > len(path)/fullDistance and len(path)/fullDistance > .75)
            print(len(path))
        print(f"Finished Cell {i}")
        img[:,:,0] = np.copy(grid[:,:])
        img[start[0], start[1], 2] = 255
        img[goal[0], goal[1], 2] = 255
        plt.imshow(img)
        plt.title(f"Cell {i} to Cell {shockwavedCell}")
        plt.show()
    paths = np.array(paths)
    minDistance = min(len(arr) for arr in paths[np.where(connectionMap == 2)])
    maxDistance = max(len(arr) for arr in paths[np.where(connectionMap == 2)])
    for i in range(len(centers)):
        if (connectionMap[i] == 2):
            connectionMap[i] = int(len(paths[i]) < ((minDistance + maxDistance)//(len(centers)//4) )) + 1
            print(len(paths[i]))
    return connectionMap