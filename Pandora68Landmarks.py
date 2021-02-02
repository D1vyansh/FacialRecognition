# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:21:31 2020

@author: Atharva
"""

from PIL import Image
from numpy import asarray
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from imutils import face_utils
from tqdm import tqdm
import numpy as np
import os
import cv2
import csv
import dlib
import imutils
import math


def interocularDist(dlib_points):
  iodList = []
  xl,yl = dlib_points.part(43).x, dlib_points.part(43).y
  iodList.append(np.array([xl,yl]))
  xr,yr = dlib_points.part(37).x, dlib_points.part(37).y
  iodList.append(np.array([xr,yr]))
  iod=np.linalg.norm(iodList[0] - iodList[1])
  if iod!=0:
    return iod
  else:
    return 1

def detect_face_points(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    face_rect = detector(image, 1)
    if len(face_rect) != 1: return []
    dlib_points = predictor(image, face_rect[0])
    iod = interocularDist(dlib_points)
    face_points = []
    for i in range(68):
        x, y = dlib_points.part(i).x, dlib_points.part(i).y
        x = (x/iod)*100
        y = (y/iod)*100
        face_points.append(x)
        face_points.append(y)
    return face_points

all_roll=[]
all_pitch = []
all_yaw = []
png_files = []
with open('/home/at2216/faceLandmark/rpy_gt_final_all.csv', mode='r') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     for row in reader:
         png_files.append(row[0])
         all_roll.append(row[1])
         all_pitch.append(row[2])
         all_yaw.append(row[3])
         
yaw = np.asarray(all_yaw)
pitch = np.asarray(all_pitch)
roll = np.asarray(all_roll)
print(png_files[2])

discard = []
index = -1
with open('./Pandora68Landmarks.csv', mode='wb') as file:  
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for f in png_files:
        index = index + 1
        print(f)
        print(index)
        im = cv2.imread(f)
        face_points = detect_face_points(im)
        print(len(face_points))
        if len(face_points) != 136:
            discard.append(index)
            continue
        #features = compute_features(face_points)
        
        writer.writerow(face_points)
        
aYaw = np.delete(yaw,discard)
aPitch = np.delete(pitch,discard)
aRoll = np.delete(roll,discard)

for ind in sorted(discard,reverse=True):
    del png_files[ind]
print(len(png_files))
print(len(aYaw))
print(png_files[100])
print(aYaw[100])

with open('./Pandora68GT.csv', mode='wb') as file:  
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)       
    for i in range(len(aYaw)):
#        yawfloat = allYaw[i]
#        strfloat = "".join(str(x) for x in yawfloat)
#        pitchfloat = allPitch[i]
#        strpitch = "".join(str(x) for x in pitchfloat)
#        rollfloat = allRoll[i]
#        strroll="".join(str(x) for x in rollfloat)
        writer.writerow([aYaw[i],aPitch[i],aRoll[i]])