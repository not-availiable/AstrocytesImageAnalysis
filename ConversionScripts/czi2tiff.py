import tifffile as tf
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

# get outputs from bash script
input_czi = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])
intendedTraining_dir = os.path.abspath(sys.argv[3])

# make the necessary directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(intendedTraining_dir, exist_ok=True)

# magic that converts .czi to .tif
czi2tif(input_czi)

print("Conversion successful, starting tif to tiff conversion", flush=True)

# Read the TIF file
with tf.TiffFile(input_czi + ".tif") as tif:
    # Iterate over each timestamp
    for i, timestamp in enumerate(tif.pages):
        # path to eventual tiff file
        output_file = os.path.join(output_dir, f"{i}.tiff")
        if i < 5:
            output_file = os.path.join(intendedTraining_dir, f"{i}.tiff")
	# create tiff
        tf.imwrite(output_file, timestamp.asarray())
        # grab newly created file
        src = cv2.imread(output_file)
        # set red and blue channles to 0
        src[:,:,0]=0
        src[:,:,2]=0
        # overwrite initial image
        cv2.imwrite(output_file, src)

print("Conversion successful")
