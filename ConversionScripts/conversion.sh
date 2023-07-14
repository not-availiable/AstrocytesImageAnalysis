#!/bin/bash
currentDir=$(pwd)
targetDir=$1
outputDir=$2
cd "$targetDir"
ls *.czi > listOfFiles.txt
mv listOfFiles.txt "$currentDir"
cd "$currentDir"
listOfFiles=()
while read -r line
do
	listOfFiles+=("$line")
done < listOfFiles.txt
for i in "${listOfFiles[@]}"
do
	fileName=$i
	inputFilePath="${targetDir}/$fileName"
	fileName=${fileName%.*}
	outputFolderPath="$outputDir/${fileName##*/}"
	intendedTrainingPath="$outputFolderPath/IntendedTrainingFiles"
	python czi2tiff.py "$inputFilePath" "$outputFolderPath" "$intendedTrainingPath"
	cd "$targetDir"
	rm *.tif
	cd "$currentDir"
done
rm listOfFiles.txt
