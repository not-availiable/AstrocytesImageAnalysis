import os

input_folder = 'input folder'
output_folder = 'output folder'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of files in the input folder
files = os.listdir(input_folder)

# Sort the filenames numerically (without leading zeros)
sorted_files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

# Calculate the number of leading zeros needed
num_zeros = len(str(len(sorted_files)))

# Rename and move the files to the output folder
for i, filename in enumerate(sorted_files):
    file_extension = os.path.splitext(filename)[1]
    new_filename = f"{str(i+1).zfill(num_zeros)}{file_extension}"
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, new_filename)
    os.rename(input_path, output_path)

print("Renaming and sorting complete!")