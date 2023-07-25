# Imports for the code
import numpy as np
import heapq
import matplotlib.pyplot as plt
import cv2
import math
from scipy import ndimage

# GLOBAL VARIABLES
connectionMap = np.zeros((1,1)) # 0 = not connected, 1 = networked, 2 = connected
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
    return reversed(path)


def a_star_search(grid, src, dest):
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
                    connectionMap[src][dest] = 2
                    connectionMap[dest][src] = 2
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
                distance = calculate_h_value(x, y, goal)
                if distance < min_distance:
                    min_distance = distance
                    nearest_square = (x, y)
    connectionMap[src][dest] = 1
    connectionMap[dest][src] = 1
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

# Reusable Functions (other than A*)
def getCellCenter(mask):
    FormerX = mask.shape[0] + 5
    LatterX = 0
    FormerY = mask.shape[1] + 5
    LatterY = 0
    maskOnly = np.copy(mask)
    for idx, obj in np.ndenumerate(maskOnly):
        if (obj == 1):
            FormerX = min(FormerX, idx[0])
            FormerY = min(FormerY, idx[1])
            LatterX = max(LatterX, idx[0])
            LatterY = max(LatterY, idx[1])
    CenterX = (FormerX + LatterX) // 2
    CenterY = (FormerY + LatterY) // 2
    return (CenterX, CenterY)

# def getCellCenter(mask):
#     indices = np.where(mask == 1)
#     FormerX = np.min(indices[0])
#     LatterX = np.max(indices[0])
#     FormerY = np.min(indices[1])
#     LatterY = np.max(indices[1])
#     CenterX = (FormerX + LatterX) // 2
#     CenterY = (FormerY + LatterY) // 2
#     return (CenterX, CenterY)

# Getting a test image (not being used in the model yet)
image = cv2.imread('/Users/connor/Downloads/TrainingSet/34_1.tiff')
mk = np.load('/Users/connor/Downloads/TrainingSet/34_1_seg.npy', allow_pickle=True).item()['masks']
img = np.copy(image)
mask = np.copy(mk)
img = cv2.resize(img, (SIZE, SIZE))
mask = cv2.resize(img, (SIZE, SIZE))
plt.imshow(img)
plt.title("Shrunken Image")
plt.show()
background = int(abs(np.mean(img) - np.median(img)))*3
print(background)
img[img < background] = 0
img[:,:,0] = 0
img[:,:,2] = 0
plt.imshow(img)
plt.title("Removed Background")
plt.show()
masks = []
centers = []
print("Removed Background")
for i in range(1, np.max(mask)+1):
    temp = np.copy(mask)
    temp[temp != i] = 0 
    temp[temp == i] = 1
    masks.append(temp)
    centers.append(getCellCenter(masks[-1]))
print("Got Masks and Centers")

# One Cell Connection Code

# start = 25
# end = 28
# grid = np.copy(img)[:,:,1]
# grid[grid > 0] = 1
# StartCenterX, StartCenterY = getCellCenter(masks[start])
# EndCenterX, EndCenterY = getCellCenter(masks[end])
# print("Got Centers")
# start = (StartCenterX, StartCenterY)
# goal = (EndCenterX, EndCenterY)
# path, isConnected = a_star_search(grid, start, goal)
# for point in path:
#     x = point[0]
#     y = point[1]
#     grid[x][y] = 255
# print("Finished")
# plt.imshow(grid)
# plt.show()
# img[:,:,0] = grid[:,:]
# img[StartCenterX, StartCenterY, 2] = 255
# img[EndCenterX, EndCenterY, 2] = 255
# plt.imshow(img)
# plt.show()

grid = np.copy(img)[:,:,1]
grid[grid > 0] = 1
print("Created Grid")
connectionMap = np.zeros((len(masks),len(masks)))
shockwavedCell = 26
print("Created Map")
for i in range(1, len(masks)+1):
    if i == shockwavedCell:
        continue
    print(f"Got Centers For Cell {i}")
    start = centers[i]
    goal = centers[shockwavedCell]
    print(centers)
    path = a_star_search(grid, start, goal)
    print(f"Got Path For Cell {i}")
    for point in path:
        x = point[0]
        y = point[1]
        grid[x][y] = 255
    print("Finished")
    img[:,:,0] = grid[:,:]
    img[start[0], start[1], 2] = 255
    img[goal[0], goal[1], 2] = 255
    plt.imshow(img)
    plt.title(f"Cell {i} to Cell {shockwavedCell}")
    plt.show()