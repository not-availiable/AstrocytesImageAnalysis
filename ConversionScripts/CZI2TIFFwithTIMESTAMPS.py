from czifile import CziFile
import xml.etree.ElementTree as ET
import os
import tifffile
import time
start_time=time.time()
def extract_timestamps_from_metadata(metadata_xml):
    root = ET.fromstring(metadata_xml)
    timestamps = [element.text for element in root.iter('Time')]
    return timestamps

def convert_czi_to_tiff(czi_file_path, output_dir):
    with CziFile(czi_file_path) as czi:
        img = czi.asarray()
        metadata_xml = czi.metadata()
        
        timestamps = extract_timestamps_from_metadata(metadata_xml)

        for i, slice_img in enumerate(img):
            timestamp = timestamps[i].replace(":", "_").replace(".", "_")
            tiff_file_name = f"{timestamp}_{i}.tiff"
            tifffile.imwrite(os.path.join(output_dir, tiff_file_name), slice_img)

czi_file = "CZI_FILE_HERE"
output_dir = "OUTPUT_DIRECTORY_HERE"

convert_czi_to_tiff(czi_file, output_dir)
end_time=time.time()
final=end_time-start_time
print(final)