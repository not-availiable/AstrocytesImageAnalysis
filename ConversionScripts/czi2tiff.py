import tifffile as tf
import czifile
import xml.etree.ElementTree as ET
import xmltodict
import numpy as np
import cv2
import os
import sys

try:
    from .czifile import czi2tif
except ImportError:
    try:
        from czifile.czifile import czi2tif
    except ImportError:
        from czifile import czi2tif

def extract_timestamps_from_metadata(metadata_xml):
    root = ET.fromstring(metadata_xml)
    timestamps = [element.text.replace(":", "-") for element in root.iter('Time')]
    return timestamps

# get outputs from bash script
input_czi = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])
intendedTraining_dir = os.path.abspath(sys.argv[3])
is_renaming = int(sys.argv[4])

# make the necessary directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(intendedTraining_dir, exist_ok=True)

czi = czifile.CziFile(input_czi)

# magic that converts .czi to .tif
if not is_renaming:
    czi2tif(input_czi)
    print("Conversion successful, starting tif to tiff conversion", flush=True)
else:
    print("Renaming files with timestamps")

# get timestamps
metadata_xml = czi.metadata()
timestamps = extract_timestamps_from_metadata(metadata_xml)


# Read the TIF file
if not is_renaming:
    with tf.TiffFile(input_czi + ".tif") as tif:
        # Iterate over each timestamp
        for i, image in enumerate(tif.pages):
            # path to eventual tiff file    
            current_output_dir = output_dir
            output_file = os.path.join(output_dir, f"{timestamps[i]}.tiff")
            if i < 5:
                current_output_dir = intendedTraining_dir
                output_file = os.path.join(intendedTraining_dir, f"{timestamps[i]}.tiff")

            # create tiff
            tf.imwrite(output_file, image.asarray())
            # grab newly created file
            src = cv2.imread(output_file)
            # set red and blue channles to 0
            src[:,:,0]=0
            src[:,:,2]=0
            # overwrite initial image
            cv2.imwrite(output_file, src)
else:
    for i, file in enumerate(os.listdir(output_dir)):
        if os.path.isfile(file):
            os.rename(os.path.join(output_dir, file), os.path.join(output_dir, f"{timestamps[i]}.tiff"))
    if output_dir != intendedTraining_dir:
        for i, file in enumerate(os.listdir(intendedTraining_dir)):
            if os.path.isfile(file):
                os.rename(os.path.join(intendedTraining_dir, file), os.path.join(intendedTraining_dir, f"{timestamps[i]}.tiff"))


# if is_renaming:
#     os.rename(output_file, os.path.join(current_output_dir, timestamps[i]))

print("Conversion successful")

