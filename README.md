# MRI_lava
Processing of MRI images of analogue lavas

## Technologies:

This code is written for Matlab 2020b and Python 3.7.

## Contains:

Matlab:
load_dicom.m - script to read a series of dicom images and save as a video.
Data  - data from an MRI experiment. Data/BEAT_FQ2 is measurement signal intensity, Data/BEAT_FQ2_P is velocity encoded in the direction of flow.

Python: 
Model/dambreak.py - depth-integrated forward model (see https://github.com/JanineBirnbaum18/3-phase-flow)
Model/lit_models.py - functions for rheology parameterizations for two- and three-phase suspensions
Model/trig_fund.py - shortcuts for trig functions in degrees
Experiments list.xlsx - Excel table of experiment data
Main.ipynb - Jupyter notebook for running EnKF inversion
Read_dicom.ipynb - Jupyter notebook for visualizing dicom data
build_mesh.py - subscript for constructing mesh from outline
enivornment.yml - description of environment requirements
gfmodel.py - subscript to run 2D FEM forward model
read_dicom.py - subscript to load dicom data

## Workflow
Run Main.ipynb to call subscripts that load dicom data and run forward model for EnKF analysis
