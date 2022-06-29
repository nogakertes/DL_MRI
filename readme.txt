This code base is Amitay Lev and Noga Kertes final project in the course DeepMRI in the Technion during spring 2022.
The code ia written in python and uses Pytorch as well as a few utilities from fastMRI git.

--------------------------------------------------- Code ---------------------------------------------------
Code files description:
1. config.py - Configuration file for defining the experiment's parameters - user, paths, hyper parameters etc.
2. baseModelTrain.py - Python script for loading fastMRI data from defined paths in config.py and training a model.
3. evaluate_oneNet.py - Python script for evaluating a model - accuracy and metrics calculations and samples plotting.
4. losses.py - Python codes containing different loss functions and utilities we used in the project.
5. data_loader.py - Data utilities for kspace data, mainly imported from fastMRI git.
6. models.py - Repository for the model we used in this project.
7. utils.py - Functions and utilities we used more than once in the project.

------------------------------------------------- Database -------------------------------------------------
In this project we used fastMRI database, specifically we used the single coil knee data and extracted part of the dataset to our directory on the server.
The files are in ".h5" format and the path to get them is described in 'config.py':
TRITON_DATA_PATH = '/home/stu1/'

---------------------------------------------- Model training ----------------------------------------------
In order to train the model, do the following steps:
1. Make sure you install the requirements described in 'requirement.txt'
2. Define specific paths and parameters in 'config.py'.
   Currently the code is ready to be launched on triton01 server.
3. Run 'baseModelTrain.py'.

--------------------------------------------- Model evaluation ----------------------------------------------
In order to generate results and evaluate the model, do the following steps:
1. Make sure you have the correct model name and paths in 'config.py'.
2. Run 'evaluate_oneNet.py'.
