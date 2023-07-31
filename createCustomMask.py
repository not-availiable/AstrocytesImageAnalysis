import sys
import os
import math
import numpy as np
import numpy.ma as ma
from cellpose import plot, utils
from matplotlib import pyplot as plt
import tifffile as tf
import cv2
def findCenterOfMask(mask):
    cytoCenterX = 0
    cytoCenterY = 0
    for x in mask[:,0]:
        cytoCenterX+=x
    for y in mask[:,1]:
        cytoCenterY += y
    cytoCenterX /= len(mask[:,0])
    cytoCenterY /= len(mask[:,1])
    return cytoCenterX,cytoCenterY
                   
def createTif(image,shockIndex,otherMaskIndex,bigMask,pathing):
    tiff = np.copy(image)
    # Create numpy arrays for the first and second channels
    first_channel = np.zeros_like(src[:,:,0])
    third_channel = np.zeros_like(src[:,:,2])
    firstMask = bigMask == shockIndex
    thirdMask = bigMask == otherMaskIndex
    first_channel[firstMask] = 1
    #second_channel[:,:] = 255
    third_channel[thirdMask] = 1
# Set the first and second channels to these arrays
    tiff[:,:,0] = first_channel
    tiff[:,:,2] = third_channel
    
    cv2.imwrite(f'{pathing}.tif', tiff)

def createPath(initialPath, classification,index,id):
    
    data_subfolder = "data"
    folders = ["unconnected", "networked", "connected"]
    for folder in folders:
    # Use os.path.join to construct the full path to the directory
        directory = os.path.join(initialPath, data_subfolder, folder)
        os.makedirs(directory, exist_ok=True)
    d = os.path.join(initialPath, data_subfolder, folders[classification],f"{index}_{id}")
    return d
    
    
#manually enable repo location, image 
if __name__ == '__main__':
    repoLocation = input("Enter current repository: ")
    imagesally = input("Enter Image repo (unique):")
    print("Running code...")
    f = open("cytoMaskLocation.txt")
    cytoDat = np.load(f.readline().rstrip(), allow_pickle=True).item()
    f.close()

    # plot image with outlines overlaid in red
   
    cytoOutlines = utils.outlines_list(cytoDat['masks'])

    f = open("imgLocation.txt")
    imagesrc = f.readline().rstrip()
    samplingImage = plt.imread(imagesrc)
    src = cv2.imread(imagesrc)
    f.close()

    masks = []


    cytoWholeMask = cytoDat['masks']

    # i = 0
    counter = 0
    for o in cytoOutlines:

        
        plt.plot(o[:,0], o[:,1], color='r')
        x,y = findCenterOfMask(o)
        plt.annotate(str(counter), (x,y), color="white")
        # plt.plot(c[:,0], c[:,1], color='r')
        
        counter+=1

        #plt.plot(centerX, centerY, color='r', marker=".")

        # cytoplasm
        # see if there is a cytoplasm that is close enough to a nucleus to use
        
        closeMaskId = 1
        

        # use only the relavant part of the cytoplasm mask 
        mask = cytoWholeMask
        
       
    # stitch together all of the masks
    fullMask = np.zeros(cytoWholeMask.shape)
    for mask in masks:
        fullMask = np.add(fullMask, mask)
    fullMask = fullMask > 0
    

    samplingImage = np.copy(samplingImage)
    print("Figure out each of the connections before closing the plot")
    print()
    print("---------------------------")
    print("Label each cell accordingly (relationship to shockwave)")
    print("0 - Not connected")
    print("1 - networked (indirectly connected)")
    print("2 - directly connected")
    print("3 - is Shockwave cell")
    print("---------------------------")
    print("Close the plot to continue")
    plt.imshow(samplingImage)
    plt.show()
    
    inputList = []
    shockwaveMaskNumber = 0
    for i in range(counter):
        w = int(input("Cell " + str(i)+ ": "))
        if w==3:
            shockwaveMaskNumber = i
        inputList.append(w)
    
   
    for index in range(counter):
        if index == shockwaveMaskNumber:
            continue
        path = createPath(repoLocation,inputList[index],index,imagesally)
        createTif(src,shockwaveMaskNumber+1,index+1,cytoWholeMask,path)
        index+=1