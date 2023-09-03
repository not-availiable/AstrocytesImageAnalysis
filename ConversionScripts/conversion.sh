#!/bin/bash
# get where the script is being run
currentDir=$(pwd)
# where the .czi files are
targetDir=$1
# where the .tiff folders will go
outputDir=$2
# 0 to create new files 1 to rename existing ones
isRenaming=$3
# get a list of the names
cd "$targetDir"
ls *.czi > listOfFiles.txt
mv listOfFiles.txt "$currentDir"
cd "$currentDir"
# add the list of names to an array
listOfFiles=()
while read -r line
do
	listOfFiles+=("$line")
done < listOfFiles.txt
for i in "${listOfFiles[@]}"
do
	# add file name to rest of path
	fileName=$i
	inputFilePath="${targetDir}/$fileName"
	# remove the file ending to make an output folder name
	fileName=${fileName%.*}
	outputFolderPath="$outputDir/${fileName##*/}"
	# path to directory that will contain the first five images
	intendedTrainingPath="$outputFolderPath/IntendedTrainingFiles"
	# run the conversion script on the target .czi
	python czi2tiff.py "$inputFilePath" "$outputFolderPath" "$intendedTrainingPath" $isRenaming
	# remove the middle conversion step
	cd "$targetDir"
	rm *.tif
	cd "$currentDir"
done
# remove the file list
rm listOfFiles.txt
