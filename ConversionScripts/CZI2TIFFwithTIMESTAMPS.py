from czifile import CziFile
import xml.etree.ElementTree as ET
import os
import tifffile
import time
import cv2
import dask

def extract_timestamps_from_metadata(metadata_xml):
    root = ET.fromstring(metadata_xml)
    timestamps = [element.text for element in root.iter('Time')]
    return timestamps

@dask.delayed
def worker(slice_img, timestamp, i, output_dir):
    tiff_file_name = f"{timestamp}_{i}.tiff"
    tifffile_path = os.path.join(output_dir, tiff_file_name)
    tifffile.imwrite(tifffile_path, slice_img)

    # Read the image back in
    img = cv2.imread(tifffile_path)

    # Set the red and blue channels to 0
    img[:,:,0] = 0
    img[:,:,2] = 0

    # Save the modified image back to the file
    cv2.imwrite(tifffile_path, img)

def convert_czi_to_tiff(czi_file_path, output_dir):
    with CziFile(czi_file_path) as czi:
        img = czi.asarray()
        metadata_xml = czi.metadata()
        timestamps = extract_timestamps_from_metadata(metadata_xml)

        tasks = []
        for i, slice_img in enumerate(img):
            timestamp = timestamps[i].replace(":", "_").replace(".", "_")
            tasks.append(worker(slice_img, timestamp, i, output_dir))

        # Execute all tasks
        dask.compute(*tasks)

if __name__ == "__main__":
    czi_file = "/media/krishna/OS/OPALSdata/czi2tiff/czi/Cell_91_Pre.czi"
    output_dir = "/media/krishna/OS/OPALSdata/czi2tiff/output"

    start_time = time.time()
    convert_czi_to_tiff(czi_file, output_dir)
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken: {:.2f} seconds".format(total_time))
