import os

# Replace 'path_to_folder' with the path to your main folder containing the subfolders with TIFF files.
path_to_folder = 'path to allcellstiffs'

# Function to add leading zeros to file names
def rename_files_with_leading_zeros(folder_path):
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.tiff'):
                file_path = os.path.join(root, file_name)
                file_num = int(file_name.split('.')[0])
                new_file_name = f"{file_num:03d}.tiff"  # Adds leading zeros to make it three digits (adjust as needed)

                if file_name != new_file_name:
                    new_file_path = os.path.join(root, new_file_name)
                    os.rename(file_path, new_file_path)
                    print(f"Renamed {file_path} to {new_file_path}")

# Call the function to add leading zeros to all TIFF files in the folders
rename_files_with_leading_zeros(path_to_folder)
