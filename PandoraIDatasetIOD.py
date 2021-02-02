# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:15:48 2020

@author: Atharva Tembe
"""

from PIL import Image
from numpy import asarray
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from imutils import face_utils
from tqdm import tqdm
import numpy as np
import os
import json
import cv2
import csv
import dlib
import imutils
import math



def getListOfGTFiles(dirName):

   listOfFiles = os.listdir(dirName)
   allContents = list()
   for e in listOfFiles:
      #full path to directory
      fp = os.path.join(dirName, e)
      if os.path.isdir(fp):
         allContents = allContents + getListOfGTFiles(fp)
      elif ((fp.find(".json") != -1) and ((fp.find("21") == -1) and ((fp.find("22") == -1)) and ((fp.find("14") == -1)) and ((fp.find("17") == -1)) and (((fp.find("12") == -1)) and (fp.find("16") == -1)) and ((fp.find("20") == -1)))):
         allContents.append(fp)
      elif ((fp.find(".txt") != -1) and ((fp.find("21") != -1) or (fp.find("14") != -1) or (fp.find("22") != -1) or (fp.find("17") != -1) or (fp.find("16") != -1) or (fp.find("12") != -1 )  or (fp.find("20") != -1))):
         allContents.append(fp)
   return allContents



def getListOfPNGFiles(dirName):
   listOfFiles = os.listdir(dirName)
   allContents = list()
   for e in listOfFiles:
      #full path to directory
      fp = os.path.join(dirName, e)
      if os.path.isdir(fp):
         allContents = allContents + getListOfPNGFiles(fp)
      elif ((fp.find("_RGB.png") != -1)):
         allContents.append(fp)
   return allContents


def rect_to_bb(rect):
   x = rect.left()
   y = rect.top()
   w = rect.right() - x
   h = rect.bottom() - y
   
   return (x, y, w, h)
   

def shape_to_np(shape, dtype="int"):
   coords = np.zeros((68, 2), dtype=dtype)
   for i in range(0, 68):
      coords[i] = (shape.part(i).x, shape.part(i).y)
   return coords





def getGTFromJson(json_data):
   yawl = list()
   pitchl = list()
   rolll = list()
   filel = list()
   numFiles = 0
   for d in json_data:
      #print(d)
      pose = d["orientation"]
      euler = pose['euler']
      fileCorr = ("0"*(6-len(str(d["frame_num"])))) + str(d["frame_num"]) + "_RGB.png"
      #print(fileCorr)
      
      filel.append(fileCorr)
      yawl.append(euler['yaw'])
      pitchl.append(euler['pitch'])
      rolll.append(euler['roll'])
      numFiles += 1
   yaw = np.asarray(yawl)
   pitch = np.asarray(pitchl)
   roll = np.asarray(rolll)
   fileP = np.asarray(filel)
   return filel, yawl, pitchl, rolll, numFiles
      





photo_width_IOD_val = 6.08    #US Passport Metric
photo_top_IOD_val = 2.94      #Malaysia Passport Metric
photo_bottom_IOD_val = 4.61   #Canadian Passport Metric



#json_data = json.load('./pandora/01/base_1_ID01/data.json')
outDir = './preprocessed_pandora/'
pandDir = './pandora/20/'
numFiles = 0
img_shape = (299, 299, 3)



allYaw = list()
allPitch = list()
allRoll = list()
allIndx = list()
allFiles = list()

json_files = getListOfGTFiles(pandDir)





print("Parsing JSON & Text files for Ground Truth Pitch, Roll, Yaw")
for f in json_files:
   if (f.find("json") != -1):
      print("!!!PARSING JSON FILES!!!")
      print(f)
      with open(f) as j:
         json_data = json.load(j)
         pand_dir = f[f.find('/pandora/') + 9:f.find('/pandora/')+11]
         fileCorr, yaw, pitch, roll, fileIts  = getGTFromJson(json_data)
         jsonDir = f[:f.find('data')]
         filepaths = [str(jsonDir) + "RGB/" +str(fl) for fl in fileCorr[::]]
         yawres = yaw[::]
         pitchres = pitch[::]
         rollres = roll[::]
         allYaw.extend(yawres)
         allPitch.extend(pitchres)
         allRoll.extend(rollres)
         allFiles.extend(filepaths)
         numFiles += fileIts
   else:
      print("!!!PARSING TEXT FILES!!!")
      print(f)
      yawl = list()
      pitchl = list()
      rolll = list()
      indxl = list()
      filepaths = list()
      pand_dir = f[f.find('/pandora/') + 9:f.find('/pandora/')+11]
      with open(f) as fl:
         for line in fl:
            numFiles += 1
            lineArr = line.split()
            yawl.append(lineArr[4])
            pitchl.append(lineArr[3])
            rolll.append(lineArr[2])
            jsonDir = f[:f.find('data')]
            fileCorr = str(jsonDir) + "RGB/" + str(lineArr[1]) + "_RGB.png"
            filepaths.append(fileCorr)
         yawres = yawl[::]
         pitchres = pitchl[::]
         rollres = rolll[::]
         indxres = indxl[::]
         fileres = filepaths[::]
      allYaw.extend(yawres)
      allPitch.extend(pitchres)
      allRoll.extend(rollres)
      allIndx.extend(indxres)
      allFiles.extend(filepaths)
face_data = np.reshape(np.zeros(numFiles * img_shape[0] * img_shape[1] * img_shape[2]), (numFiles, img_shape[0], img_shape[1], img_shape[2]))


newYaw = list()
newPitch = list()
newRoll = list()


indx = 0
fileC = 0
fileNum = 0
eye = 1
#variables for debugging
rsz = 1
sv = 1
post_rsz = 0
print("Parsing Image Files into Bounding Box Face")
for p in allFiles:
   img_raw = cv2.imread(p)
   im = Image.open(p).convert("RGB")
   oW, oH = im.size
   gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
   face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')  
   eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
   face_rects = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
   print(' Faces Found: ', len(face_rects))
   if len(face_rects) == 1:
      if (eye == 0):
         newYaw.append(allYaw[fileC])
         newPitch.append(allPitch[fileC])
         newRoll.append(allRoll[fileC])
         fileNum += 1
      else:
         for x, y, w, h in face_rects:
            eyes = eye_cascade.detectMultiScale(gray[y:y+h,x:x+w])
            newimg = Image.fromarray(gray[y:y+h, x:x+w])
         
            if len(eyes) == 2 and eye == 1:
               print(len(eyes), " Eyes found at index ", indx)
               centerx = np.reshape(np.zeros(2), (2))
               centery = np.reshape(np.zeros(2), (2))
               face_center = np.reshape(np.zeros(2, dtype = int), (2))
               centerh = np.reshape(np.zeros(2), (2))
               i = 0
               for (ex, ey, ew, eh) in eyes:
                  centerx[i] = ex + ew // 2
                  centery[i] = ey + ey // 2
                  i += 1
               #calculate inter oculary distance between two eyes for cropping
               distx = abs(centerx[0] - centerx[1])
               disty = abs(centery[0] - centery[1])
               iod = math.sqrt(distx**2 + disty**2)
            
               #determine center of image to build cropping around
               if (centerx[0] > centerx[1]):  
                  face_center[0] = centerx[1] + distx // 2
               else:
                  face_center[0] = centerx[0] + distx // 2
               if (centery[0] > centery[1]):
                  face_center[1] = centery[1] + disty // 2
               else:
                  face_center[1] = centery[0] + disty // 2
             
               center_imgx = face_center[0] + x
               center_imgy = face_center[1] + y
            
               tl_x = int(round(center_imgx - photo_width_IOD_val * iod / 2))
               tl_y = int(round(center_imgy - photo_top_IOD_val * iod))
               crop_w = int(round(photo_width_IOD_val * iod))
               crop_h = int(round((photo_top_IOD_val + photo_bottom_IOD_val) * iod))
            
            
            
            
               newImCrop = im.crop((tl_x,                      #top left x coordinate
                                    tl_y,                      #top left y coordinate 
                                    tl_x+crop_w,                    #width of crop
                                    tl_y+crop_h))                   #height of crop
            
               if (sv):
                  print("index ", fileC, " has iod = ", iod, "\t and center", face_center)
                  newImCrop.save(outDir + str(p[p.find("/pandora/") + 12:p.find("/RGB/")]) + str(p[p.find("/pandora/")+9:p.find("/pandora/")+11]) + "-" + str(p[p.find("/RGB/")+5:]))
                  
               if (rsz):
                  newImCrop = newImCrop.resize((img_shape[0], img_shape[1]))
            
               face_data[indx] = np.array(newImCrop)
               print("file: ", str(p), ":\n\tYaw: ", str(allYaw[fileC]), "\n\tPitch: ", str(allPitch[fileC]), "\n\tRoll: ", str(allRoll[fileC]))
               
               newYaw.append(allYaw[fileC])
               newPitch.append(allPitch[fileC])
               newRoll.append(allRoll[fileC])
               fileNum += 1 
            elif len(eyes) != 2 and eye == 1:
               print("No single pair of eyes found at index ", fileC, ". Eyes found = ", len(eyes))
               numFiles -= 1
               indx -= 1
            #if(indx >= 10):
            #   break;
   else:
      #no faces found
      print("No face / Multiple Faces found at index ", indx)
      numFiles -= 1
      indx -= 1
   indx += 1
   fileC+=1

face_data = face_data[0:fileNum,:,:,:]

yaw = np.asarray(newYaw)
pitch = np.asarray(newPitch)
roll = np.asarray(newRoll)


newSize = (299, 299, 3)
yGT = np.reshape(np.zeros(len(yaw) * 3), (len(yaw),3))
yGT = np.stack((roll, pitch, yaw), axis = 1)

AllData = np.asarray(face_data)


print("Ground Truth Samples: ", yGT.shape)
print("Data Samples: ", AllData.shape)





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


def getListOfPNGFiles(dirName):
    listOfFiles = os.listdir(dirName)
    allContents = list()
    for e in listOfFiles:
      fp = os.path.join(dirName, e)
      if os.path.isdir(fp):
         allContents = allContents + getListOfPNGFiles(fp)
      elif ((fp.find("_RGB.png") != -1)):
         allContents.append(fp)
    return allContents



png_files = getListOfPNGFiles(outDir)
print(len(png_files))

print(png_files[2])
print(yaw[2])

discard = []
index = -1
with open('./PandoraIODLandmarks.csv', mode='wb') as file:  
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for f in png_files:
        index = index + 1
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

with open('./PandoraIODGT.csv', mode='wb') as file:  
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)       
    for i in range(len(aYaw)):
#        yawfloat = allYaw[i]
#        strfloat = "".join(str(x) for x in yawfloat)
#        pitchfloat = allPitch[i]
#        strpitch = "".join(str(x) for x in pitchfloat)
#        rollfloat = allRoll[i]
#        strroll="".join(str(x) for x in rollfloat)
        writer.writerow([png_files[i],aYaw[i],aPitch[i],aRoll[i]])
        
