#!/bin/bash

# Set the paths to the source and destination folders
sourceFolder="/Volumes/UCSD Backup/UCI project Images"
destinationFolder="/Users/genechang/Desktop/salsa files"

# Find all .czi files in the source folder and loop through them
find "$sourceFolder" -type f -name "*.czi" | while read -r cziFile; do
    # Get the base name of the .czi file (without extension)
    cziName=$(basename "$cziFile" .czi)
    
    # Check if the corresponding folder exists in the destination
    correspondingFolder=$(find "/Users/genechang/Desktop/All Cell Tiffs with leading zeros" -type d -name "$cziName")
    
    # If the folder exists, copy it to the destination
    if [ -n "$correspondingFolder" ]; then
        cp -r "$correspondingFolder" "$destinationFolder"
    fi
done
