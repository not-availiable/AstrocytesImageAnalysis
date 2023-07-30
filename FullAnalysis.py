# Imports for the code
import pandas as pd
import numpy as np
from cellpose import models, utils
import math

# GLOBAL VARIABLES

CONNECTION_DICT = {0:"Not Connected", 1:"Networked", 2:"Connected", 3:"Dead Cell"}
stats = {}
centers = []

# Methods to make CSV
def create_dataframe(rois):
    global stats
    stats = {
        'Cell_Size': np.zeros(rois),
        'Peak_Value': np.zeros(rois),
        'Peak_Location': np.zeros(rois),
        'FWHM': np.zeros(rois),
        'FWHM_Left_Index': np.zeros(rois),
        'FWHM_Right_Index': np.zeros(rois),
        'Connection': np.zeros(rois),
        'Distance': np.zeros(rois),
        'Integral': np.zeros(rois)
    }

# Code to create the csv from the data
def create_csv():
    statsdf = pd.DataFrame(stats)
    statsdf.to_csv('AstrocyteStats.csv', index = False)

# Line intersection for FWHM
def FWHM(x, y, roi):
    # Get half the max of y
    half_max = (np.max(y) + np.min(y)) / 2.0
    # Find the left and right indices where the peak crosses half_max
    left_idx = None
    right_idx = None
    for idx in range(1, len(y)):
        if y[idx - 1] <= half_max < y[idx]:
            left_idx = idx - 1 + (half_max - y[idx - 1]) / (y[idx] - y[idx - 1])
        if y[idx - 1] >= half_max > y[idx]:
            right_idx = idx - 1 + (half_max - y[idx - 1]) / (y[idx] - y[idx - 1])
        if left_idx is not None and right_idx is not None:
            break
    # Incase there is no intersection, set to max/min of the x
    if (left_idx is None):
        left_idx = 0
    if (right_idx is None):
        right_idx = x[-1]
    # Calculate the FWHM
    fwhm = (x[int(right_idx)] + math.abs(x[int(right_idx)] - x[math.ceil(right_idx)]) * (right_idx % 1)) - (x[int(left_idx)] + math.abs(x[int(left_idx)] - x[math.ceil(left_idx)]) * (left_idx % 1))
    # Write to the data
    stats['FWHM'][roi] = fwhm
    stats['FWHM_Left_Index'][roi] = left_idx
    stats['FWHM_Right_Index'][roi] = right_idx
    
# Code to write peak data to stats
def peak(x, y, roi):
    stats['Peak_Value'][roi] = np.max(y)
    stats['Peak_Location'][roi] = np.argmax(y)
    
# Code to write cell data to stats
def cell(nucDat, roi, shock):
    nucWholeMask = np.copy(nucDat)
    stats['Cell_Size'][roi] = np.sum(nucWholeMask == roi)
    print(np.sum(nucWholeMask == roi))
    stats['Distance'][roi] = math.dist(centers[roi], centers[shock])

# Code to get centers
def getCenters(nucDat):
    global centers
    centers = []
    nucOutlines = utils.outlines_list(nucDat)
    for outline in nucOutlines:
        centers.append((int(outline[:, 0].mean()), int(outline[:, 1].mean())))
        
# Integral
def integral(x, y, roi):
    stats['Integral'][roi] = np.trapz(y, x)

def connections(roi, connectionMap):
    stats['Connection'][roi] = CONNECTION_DICT[connectionMap[roi]]

# Code to get stats
def getStats(nucDat, rois, x, ally, shockwavedCell):
    create_dataframe(rois)
    getCenters(nucDat)
    for roi in range(rois):
        FWHM(x, ally[roi], roi)
        peak(x, ally[roi], roi)
        cell(nucDat, roi, shockwavedCell)
        integral(x, ally[roi], roi)
    create_csv()