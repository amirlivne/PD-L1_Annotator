# Annotation Tool
The annotation tool is designed to assist a patologist to easily annotate H&E images according to their matching PD-L1 stained images. We supply a small dataset to test and validate the tool. The complete data can be found on http://bliss.gpec.ubc.ca/ .

# How to use
To activate the tool, run the script annotation_tool.py. The input to the script is an excel sheet that containes a list of the relevant images, and the root folder of the images. You can use the script with the default parametres attached in this code, or create different excel files to annotate your own images.

An example of use:

_python annotation_tool.py --excel_file metadata/annotation_task.xlsx --root_images_dir data_

The annotations are saved in the excel file, at the matching column. 

The tool is controled by using the following keys:

_keys 0-9_: assign to the image the decided labels and color thw window with matching color.

_enter_: save the label and move to the next example

_space_: toggle between H&E, PD-L1 and PD-1 corresponding images.

_'<'_: move backward

_'>'_: move forward

_Esc_: save results to file and exit.

The response time of the software should be immidiate on every modern CPU.

# Software requirements
This code mainly depends on the following public Python packages:

pandas 1.1.5 and above.

cv2 4.2.0.34

openpyxl 3.0.5 and above


## Installation commands
1. pip install pandas
2. pip install opencv-python==4.2.0.34
3. pip install openpyxl

## Operating systems.
This tool was tested on both Windows10 and Unbuntu 16.04 OS, using Python 3.8.
