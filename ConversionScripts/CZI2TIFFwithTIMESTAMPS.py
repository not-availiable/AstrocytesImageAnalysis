from czifile import CziFile
import xml.etree.ElementTree as ET
import os
import cv2
import tifffile as tf
import time
from multiprocessing import Pool

def extract_timestamps_from_metadata(metadata_xml):
    root = ET.fromstring(metadata_xml)
    timestamps = [element.text for element in root.iter('Time')]
    return timestamps

def save_slice_as_tiff(args):
    slice_img, timestamp, i, output_dir = args
    tiff_file_name = f"{timestamp}_{i}.tiff"
    tf.imwrite(os.path.join(output_dir, tiff_file_name), slice_img)
    src = cv2.imread(os.path.join(output_dir, tiff_file_name))
    src[:,:,0] = 0
    src[:,:,2] = 0
    cv2.imwrite(os.path.join(output_dir, tiff_file_name), src)

def convert_czi_to_tiff(czi_file_path, output_dir):
    with CziFile(czi_file_path) as czi:
        img = czi.asarray()
        metadata_xml = czi.metadata()

        timestamps = extract_timestamps_from_metadata(metadata_xml)

        with Pool() as pool:
            pool.map(save_slice_as_tiff, [(slice_img, timestamps[i].replace(":", "_").replace(".", "_"), i, output_dir) for i, slice_img in enumerate(img)])

if __name__ == "__main__":
    czi_file = "/media/krishna/OS/OPALSdata/czi2tiff/czi/Cell_91_Pre.czi"
    output_dir = "/media/krishna/OS/OPALSdata/czi2tiff/output"

    start_time = time.time()
    convert_czi_to_tiff(czi_file, output_dir)
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken: {:.2f} seconds".format(total_time))
