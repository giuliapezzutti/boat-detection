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

For the project building it is necessary to move to the project folder and: 

```
mkdir build && cd build
cmake ..
make 
```

The project is thought to be execute to predict or evaluate the performances of the pre-trained network, but it can be also used 
to generate the dataset for the training (in particular, see the next subsection). For this reason, three main types of input are accepted:

  - training (*t*): the neural network is not yet available and the images and masks folder paths are passed to generate the training dataset.
  ``` 
  ./boat-detection <images path> -t --masks=<masks path>
  ```
  
  - evaluation (*e*): the neural network is available and is tested for a dataset for which the ground truth is known; it is so necessary to provide the both the images and masks folder paths.
  ``` 
  ./boat-detection <images path> -e --masks=<masks path>
  ```
  
  - prediction (*p*): the neural network is available and is used to predict the boat position in new data, for which the masks are not known; in this case, only the images folder path is necessary. 
  ``` 
  ./boat-detection <images path> -p
  ```

Note that for the image input, it is possible to provide the folder path (all the .png images within it will be considered) or the path to a specific image. In addition, 
a image is considered a mask of the other if it has the same name and belongs to the masks folder.

To print the help string for the input, it is sufficient to execute: 
``` 
./boat-detection -h
```
  
### Replication of the training

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
./boat-detection <images path> -t --masks=<masks path>
```

Two new folders, respectively inside the images path and masks path, will be created with the correspondent saved files. The path to these 
folders must now be passed to training.py in order to perform the neural network training. 

```
python src/training.py <new images path> <new masks path>
```

The trained network is then saved in models folder. it is possible now to run main.cpp file in prediction or evaluation modes, as presented above 
(note that the re-building of the project is necessary for the correct execution!). 
