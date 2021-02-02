# KA Head pose understanding

Head yaw, pitch, and roll estimation code using a Multilayer perceptron model. Python GUI for correct head ypr annotation.
AnnotatePandora.py - Python GUI for manual head pose annotation.
MLPEval.py - Read the image and evaluate against the MLP model
Pandora68Landmarks.py - Find the 68 facial point features from each image and deduce the 2278 euclidean distance features among them.
PandoraIDatasetIOD.py - Crop the images across the IOD and compute the features