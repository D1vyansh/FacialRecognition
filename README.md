# Facial yaw, pitch, and roll understanding

## Requirements :

- Tenserflow
- Keras
- tkinter
- Pillow
- dlib
- Opencv2
- Numpy
- Matplotlib
- scikit-learn

Head yaw, pitch, and roll estimation code using a Multilayer perceptron model. Python GUI for correct head ypr annotation.
AnnotatePandora.py - Python GUI for manual head pose annotation.
MLPEval.py - Read the image and evaluate against the MLP model
Pandora68Landmarks.py - Find the 68 facial point features from each image and deduce the 2278 euclidean distance features among them.
PandoraIDatasetIOD.py - Crop the images across the IOD and compute the features

## External Dependencies :

- haarcascade_frontalface_default.xml
- haarcascade_eye.xml
- shape_predictor_68_face_landmarks.dat

## Datasets :
 
- Pandora dataset
- CelebA dataset
- Biwi dataset
- AFLW dataset

## Python Files details :

AnnotatePandora.py - This file is used to create a Python GUI for labelling the Ground Truth values of yaw, pitch and roll and saving the values in an excel 
file.

Pandora68Landmarks - This file finds the 68 facial keypoints from each input image using Dlib and normalize each keypoint by dividing them by the interocular 
distance between the two eye points. Then we deduce the 2278 euclidean distance features among them using the np.linalg.norm() function. These features are
saved in another excel file then.

PandoraIDatasetIOD.py - This file is used to crop the images according to US, Malaysia and Canada passport metric. It is used in parsing the pandora dataset
and extracting the GT values from JSON files of pandora dataset. The image files are parsed to bounding box face.

PandoraMLP.py - The multilayer perceptron model is evaluated in it. Two excels are required to run this file, the input excel containing rows of 2278 facial 
features for each input image and the GT excel containing yaw, pitch and roll columns. Hidden layers and Hyperparameters can be updated according to 
accuracy of the model. Train, validation and test split can be done using the train_test_split() function. Mean squared error loss is used in model which can
also be changed according to need. Finally, we are calculating the average yaw, pitch and roll error.

## Usage - 
CMD : python filename.py
Pycharm : Shift + F10
