# AstrocytesImageAnalysis
# This github repo contains code relating to the UCSD OPALS astrocyte project by group 15.
(the readme will look prettier later I promise)
All code is for python but different OS systems may need to adjust accordingly. 

  Support for Linux, Windows and Mac is the goal for all the code but may not be the case depending on when you are viewing the code.

Code mainly meant to be used for Astrocyte analysis as the name suggests and may not be effective on other cells.

The conversion scripts folder contains code allows for CZI files to be converted to TIFF files.

  This is in order to be easily used by cellpose and other models that are used/going to be used by us + tiff is just better for us.

AstroCytesAnalysis.py is a python script that analyses an astrocyte. 

  The script has been recently updated to be more modular and the functions can be called as needed.
    
  NucleiMaskLocation.txt provides the directory to the location of the numpy segmentation array to be used (cellpose).

  NucleiMaskLocation.txt gets called during the main astrocyte analysis script and is required for the script to work

  Same for CytoMaskLocation.txt

  Further documentation of the purpose of the code as available in #... in the python script.

  Consult the docstrings in the script for further understanding of the code.
