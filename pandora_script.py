# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:52:46 2020

@author: Atharva Tembe
"""
#from PIL import Image
from numpy import asarray
import numpy as np
import os
import json
import cv2
import csv
#import dlib
#
#
#def interocularDist(dlib_points):
#  iodList = []
#  xl,yl = dlib_points.part(43).x, dlib_points.part(43).y
#  iodList.append(np.array([xl,yl]))
#  xr,yr = dlib_points.part(37).x, dlib_points.part(37).y
#  iodList.append(np.array([xr,yr]))
#  iod=np.linalg.norm(iodList[0] - iodList[1])
#  if iod!=0:
#    return iod
#  else:
#    return 1
#
#
#def detect_face_points(image):
#    detector = dlib.get_frontal_face_detector()
#    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
#    face_rect = detector(image, 1)
#    if len(face_rect) != 1: return []
#    dlib_points = predictor(image, face_rect[0])
#    iod = interocularDist(dlib_points)
#    face_points = []
#    for i in range(68):
#        x, y = dlib_points.part(i).x, dlib_points.part(i).y
#        x = (x/iod)*100
#        y = (y/iod)*100
#        face_points.append(np.array([x, y]))
#    return face_points
#        
#
#def compute_features(face_points):
#    assert (len(face_points) == 68), "len(face_points) must be 68"
#    face_points = np.array(face_points)
#    features = []
#    for i in range(68):
#        for j in range(i+1, 68):
#            features.append(np.linalg.norm(face_points[i]-face_points[j]))
#    return features        
#    #return np.array(features).reshape(1, -1)


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


pandDir = './pandora/'
png_files = getListOfPNGFiles(pandDir)

#discard = []
#index = -1
#with open('./PandoraLandmarks.csv', mode='wb') as file:  
#    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#    for f in png_files:
#        index = index + 1
#        print(index)
#        im = cv2.imread(f)
#        face_points = detect_face_points(im)
#        if len(face_points) != 68:
#            discard.append(index)
#            continue
#        features = compute_features(face_points)
#        writer.writerow(features)
    
    
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



def getGTFromJson(json_data):
    yawl = list()
    pitchl = list()
    rolll = list()
    numFiles = 0
    for d in json_data:
        pose = d["orientation"]
        euler = pose['euler']
        yawl.append(euler['yaw'])
        pitchl.append(euler['pitch'])
        rolll.append(euler['roll'])
        numFiles += 1
    yaw = np.asarray(yawl)
    pitch = np.asarray(pitchl)
    roll = np.asarray(rolll)
    return yawl, pitchl, rolll, numFiles


numFiles = 0

allYaw = list()
allPitch = list()
allRoll = list()

json_files = getListOfGTFiles(pandDir)


print(len(json_files))
print(len(png_files))

#print("Parsing JSON files for Ground Truth Pitch, Roll, Yaw")
for f in json_files:
    if (f.find("json") != -1):
        print('In json')
        print(f)
        with open(f) as j:
            json_data = json.load(j)
            pand_dir = f[f.find('/pandora/') + 9:f.find('/pandora/')+11]
            yaw, pitch, roll, fileIts  = getGTFromJson(json_data)
            yawres = yaw[::-1] 
            pitchres = pitch[::-1] 
            rollres = roll[::-1] 
            allYaw.extend(yawres)
            allPitch.extend(pitchres)
            allRoll.extend(rollres)
            numFiles += fileIts
    else:
        yawl = list()
        pitchl = list()
        rolll = list()
        pand_dir = f[f.find('/pandora/') + 9:f.find('/pandora/')+11]
        with open(f) as fl:
            for line in fl:
                numFiles += 1
                lineArr = line.split()
                yawl.append(lineArr[4])
                pitchl.append(lineArr[3])
                rolll.append(lineArr[2])
            yawlres = yawl[::-1] 
            pitchlres = pitchl[::-1] 
            rolllres = rolll[::-1]
        allYaw.extend(yawlres)
        allPitch.extend(pitchlres)
        allRoll.extend(rolllres)
print(len(allYaw))

print(allYaw[12345])
print(png_files[12345])


#discard = [29, 53, 54, 56, 58, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 81, 82, 83,84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 132, 133, 134,135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150,151, 190, 191, 192, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221,222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,238, 239, 240, 243, 244, 245, 246, 247, 260, 261, 262, 263, 264, 265, 266, 267,268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283,284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 299, 300,301, 302, 303, 304, 307, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,350, 351, 352, 353, 354, 355, 356, 398, 399, 400, 401, 402, 403, 404, 405, 406,407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422,423, 424, 425, 426, 427, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474,475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490,491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,507, 508, 509, 510, 511, 512, 514, 515, 516, 517, 657, 659, 660, 661, 662, 663,664, 665, 666, 667, 668, 669, 670, 671, 673, 684, 685, 686, 687, 688, 689, 690,691, 698, 712, 713, 716, 720, 721, 723, 724, 725, 735, 736, 737, 738, 739, 740,741, 772, 773, 774, 775, 776, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115,1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1125, 1127, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145,1146, 1147, 1148, 1149, 1150, 1151, 1165, 1166, 1167, 1168, 1169, 1170, 1171,1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198,1199, 1200, 1203, 1204, 1206, 1207, 1209, 1210, 1211, 1212, 1214, 1217, 1218,1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1252, 1253, 1254,1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267,1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344,1345, 1346, 1347, 1348, 1349, 1350, 1351, 1393, 1394, 1395, 1396, 1397, 1398,1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425,1426, 1427, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488,1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501,1502, 1503, 1504, 1505, 1506, 1507, 1508, 1539, 1540, 1541, 1542, 1543, 1544, 1545,1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558,1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1825, 1826, 1827, 1828, 1829, 18,30, 1831, 1832, 1833, 1834, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882]
#print(len(discard))
#print(discard)
#for ind in sorted(discard,reverse=True):
#    del allYaw[ind]
#    del allPitch[ind]
#    del allRoll[ind]
#print(len(allYaw))
#with open('./PandoraGT.csv', mode='wb') as file:  
#    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)       
#    for i in range(len(allYaw)):
##        yawfloat = allYaw[i]
##        strfloat = "".join(str(x) for x in yawfloat)
##        pitchfloat = allPitch[i]
##        strpitch = "".join(str(x) for x in pitchfloat)
##        rollfloat = allRoll[i]
##        strroll="".join(str(x) for x in rollfloat)
#        writer.writerow([allYaw[i],allPitch[i],allRoll[i]])
            

