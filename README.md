# Boat Detection

## Introduction 

Automated analysis of traffic scenes mainly focuses on road scenes. However, similar methods can also be applied in a 
different scenario: boat traffic in the sea. The goal of this project is to develop a system capable of detecting boats in the image.
You should consider that the sea surface is not regular, and it is moving (and could lead to white trails), and the 
appearance of a boat can change sensibly from model to model, and depending on the viewpoint. Boat detection can be 
tackled in several different ways: the ultimate goal is to locate the boat in the image (e.g., drawing a rectangle around 
the boat). Consider that the boat can appear at any scale. For boat detection, the Intersection over Union (IoU) metric
can be exploited for the performances evaluation.

For this purpose, a deep convolutional neural network has been trained in order to perform boat detection. To have good results, 
input image and mask preprocessing and prediction processing are required and here reported. 

## Project Structure

  - Data folder can include the data used for the analysis. It is not necessary since the input folders path are passed as input.
  - Include folder contains the .h files related to the implemented c++ code.
  - Model folder includes the .pb file where the trained neural network has been saved.
  - Src folder contains all the implemented code. In particular:
    + main.cpp is the application of the BoatDetector class;
    + BoatDetector.cpp is the class containing members and methods for the boat detection computation;
    + utilities.cpp contains a set of functions useful for the computation;
    + training.py is the Python file necessary for the training of the neural network;
    + xml_to_txt.py is the Python script to convert XML files associated to the image labelling to txt ones.

## How to execute

### Completely replicate the execution

Starting from a set of images, it is necessary to label them, generating the associated XML file containing the coordinates of
top-left and down-right corner (it can be done for example with the following tool: https://github.com/tzutalin/labelImg). Move 
now to the project folder. 
The XML files must then be converted into txt with the execution of xml_to_txt.py:

```
python src/utilities/xml_to_txt.py
```

Note that eventual images containing no boats will not have the correspondent txt file. If you want to consider them in the dataset,
it is necessary to add an empty txt file with the same file name of the image in the folder in which all the files have been saved.

Subsequently, it is necessary to create the whole pre-processed dataset. To do this, the main.cpp file can be called in 'training' mode:

```
mkdir build && cd build
cmake ..
make 
./boat-detector <path image> -t --masks=<path masks>
```
