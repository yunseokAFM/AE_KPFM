Project Name

Autonomous high-throughput measurement and characterization of 2D materials using scanning probe microscopy

============

Description

This automated AFM pipeline reduced analysis time by massively compared to manual
approaches, exapnding its scalability for diverse 2D materials and various AFM imaging
modalities. 


TABLE OF CONTENTS
-----------------
1. Features
2. Requirements
3. Contact


1. FEATURES
-----------
- src : Folder containing functions necessary for Smart Remote operation 
- wheel : Folder containing python installation files for Tiff file reading and smart remote operation
- tiff : Folder containing codes necessary for reading Tiff files
- control : Folder containing functions that dictates AFM movements
- output : Folder containing Mask R-CNN training result
- object : train, validation, test annotated samples 
- Segmentation.py : Code containing Mask R-CNN procedure for segmentation 
- Smart_Remote_Launching.py : Code containing smart remote operation for automated 2D material scanning

2. REQUIREMENTS
---------------
List any prerequisites needed to run this project:
- Python 3.10
- Pandas
- shutil
- tiffread
- Pillow
- Matplotlib
- Detectron2
- numpy
- pytorch
- cuda

2.1. PSPYLIB & SmartRemote (Proprietary)
This project relies on two proprietary components supplied by Park Systems Inc.: the `pspylib` Python library and the `SmartRemote.py` script. These files are **not** included in this repository.

3. Contact
--------
https://sites.google.com/view/yunseokkim
