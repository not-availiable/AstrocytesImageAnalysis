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

def lerp(start, end, t):
    return start * (1 - t) + end * t

# Line intersection for FWHM
def yatx(x, y):
    y_lower = y[math.floor(x)]
    y_upper = y[math.ceil(x)]
    decimal = x % 1
    return lerp(y_lower, y_upper, decimal)
    

def FWHM(x, y, roi):
    # Find the maximum y-value and its index
    max_y = np.max(y)
    min_y = np.min(y)
    max_idx = np.argmax(y)
    # Calculate the half maximum value
    half_max = lerp(min_y, max_y, .5)
    left_index = max_idx
    upper_bound = max_idx
    lower_bound = 0
    while abs(yatx(left_index, y) - half_max) > .01:
        left_index = lerp(lower_bound, upper_bound, .5)
        if yatx(left_index, y) > half_max:
            upper_bound = left_index
        else: 
            lower_bound = left_index
            
    left_index = max_idx
    upper_bound = x[len(x)-1]
    lower_bound = max_idx
    right_index = max_idx
    while abs(yatx(right_index, y) - half_max) > .01:
        right_index = lerp(lower_bound, upper_bound, .5)
        if yatx(right_index, y) > half_max:
            upper_bound = right_index
        else: 
            lower_bound = right_index
        
    
    
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