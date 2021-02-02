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
        face_points.append(np.array([x, y]))
    return face_points
        

def compute_features(face_points):
    assert (len(face_points) == 68), "len(face_points) must be 68"
    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i+1, 68):
            features.append(np.linalg.norm(face_points[i]-face_points[j]))
    return features        
    #return np.array(features).reshape(1, -1)


all_roll=[]
all_pitch = []
all_yaw = []
png_files = []
with open('/home/at2216/faceLandmark/my_gt.csv', mode='r') as csvfile:
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
with open('./CelebLandmarks.csv', mode='wb') as file:  
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for f in png_files:
        index = index + 1
        print(f)
        print(index)
        im = cv2.imread(f)
        face_points = detect_face_points(im)
        print(len(face_points))
        if len(face_points) != 68:
            discard.append(index)
            continue
        features = compute_features(face_points)
        
        writer.writerow(features)
        
aYaw = np.delete(yaw,discard)
aPitch = np.delete(pitch,discard)
aRoll = np.delete(roll,discard)

for ind in sorted(discard,reverse=True):
    del png_files[ind]
print(len(png_files))
print(len(aYaw))
print(png_files[100])
print(aYaw[100])

with open('./Celeb_gt.csv', mode='wb') as file:  
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)       
    for i in range(len(aYaw)):
#        yawfloat = allYaw[i]
#        strfloat = "".join(str(x) for x in yawfloat)
#        pitchfloat = allPitch[i]
#        strpitch = "".join(str(x) for x in pitchfloat)
#        rollfloat = allRoll[i]
#        strroll="".join(str(x) for x in rollfloat)
        writer.writerow([aYaw[i],aPitch[i],aRoll[i]])
