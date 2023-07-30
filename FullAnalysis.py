# Imports for the code
import pandas as pd
import numpy as np
from cellpose import models, utils
import math

# GLOBAL VARIABLES

CONNECTION_DICT = {0:"Not Connected", 1:"Networked", 2:"Connected", 3:"Dead Cell"}
stats = {}
centers = []

# Methods to make csv
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
    # Find the maximum y-value and its index
    max_y = np.max(y)
    max_idx = np.argmax(y)
    # Calculate the half maximum value
    half_max = max_y / 2
    # Find the index of the nearest values to the left and right of the half maximum
    left_idx = np.argmin(np.abs(y[:max_idx] - half_max))
    right_idx = max_idx + np.argmin(np.abs(y[max_idx:] - half_max))
    # Calculate the FWHM
    fwhm = x[right_idx] - x[left_idx]
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
        integral(x, ally[roi])
    create_csv()