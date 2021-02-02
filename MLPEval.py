import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
from keras.callbacks import ReduceLROnPlateau


with open('./PandoraFinalLandmarks.csv', mode='r') as csvfile:
    x = list(csv.reader(csvfile))
x = np.array(x, dtype=np.float)
with open('./PandoraFinalGT.csv', mode='r') as csvfile:
    gt = list(csv.reader(csvfile))
y = np.array(gt, dtype=np.float)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)
std = StandardScaler()
std.fit(x_train)

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
        #face_points.append(x)
        #face_points.append(y)
    return face_points
    #return np.array(face_points).reshape(1, -1)
        
def compute_features(face_points):
    assert (len(face_points) == 68), "len(face_points) must be 68"
    
    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i+1, 68):
            features.append(np.linalg.norm(face_points[i]-face_points[j]))
    #return features        
    return np.array(features).reshape(1, -1)


model = load_model('./Final1panmodel.h5')

im = cv2.imread('./MLPTest/33_cropped.png', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)
features = compute_features(face_points)
features = std.transform(features)
#features = std.transform(face_points)
y_pred = model.predict(features)
yaw_pred, pitch_pred, roll_pred = y_pred[0]
print('33_cropped')
print('Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('Yaw: {:.2f}°'.format(yaw_pred))



im = cv2.imread('./MLPTest/659_cropped.png', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)
features = compute_features(face_points)
features = std.transform(features)
y_pred = model.predict(features)
yaw_pred, pitch_pred, roll_pred = y_pred[0]
print('659_cropped')
print('Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('Yaw: {:.2f}°'.format(yaw_pred))


im = cv2.imread('./MLPTest/691_cropped.png', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)
features = compute_features(face_points)
features = std.transform(features)
y_pred = model.predict(features)
yaw_pred, pitch_pred, roll_pred = y_pred[0]
print('691_cropped')
print('Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('Yaw: {:.2f}°'.format(yaw_pred))


im = cv2.imread('./MLPTest/p15.jpg', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)
features = compute_features(face_points)
features = std.transform(features)
y_pred = model.predict(features)
yaw_pred, pitch_pred, roll_pred = y_pred[0]
print('p15')
print('Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('Yaw: {:.2f}°'.format(yaw_pred))


im = cv2.imread('./MLPTest/317_cropped.png', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)
features = compute_features(face_points)
features = std.transform(features)
y_pred = model.predict(features)
yaw_pred, pitch_pred, roll_pred = y_pred[0]
print('317_cropped')
print('Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('Yaw: {:.2f}°'.format(yaw_pred))


im = cv2.imread('./MLPTest/p19_new.jpg', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)
features = compute_features(face_points)
features = std.transform(features)
y_pred = model.predict(features)
yaw_pred, pitch_pred, roll_pred = y_pred[0]
print('p19_new')
print('Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('Yaw: {:.2f}°'.format(yaw_pred))  


im = cv2.imread('./MLPTest/90_cropped.png', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)
features = compute_features(face_points)
features = std.transform(features)
y_pred = model.predict(features)
yaw_pred, pitch_pred, roll_pred = y_pred[0]
print('90_cropped')
print('Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('Yaw: {:.2f}°'.format(yaw_pred))


im = cv2.imread('./MLPTest/p2.jpg', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)
features = compute_features(face_points)
features = std.transform(features)
y_pred = model.predict(features)
yaw_pred, pitch_pred, roll_pred = y_pred[0]
print('p2')
print('Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('Yaw: {:.2f}°'.format(yaw_pred)) 


im = cv2.imread('./MLPTest/423_cropped.png', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)
features = compute_features(face_points)
features = std.transform(features)
y_pred = model.predict(features)
yaw_pred, pitch_pred, roll_pred = y_pred[0]
print('423_cropped')
print('Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('Yaw: {:.2f}°'.format(yaw_pred)) 


    


im = cv2.imread('./MLPTest/110_cropped.png', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)
features = compute_features(face_points)
features = std.transform(features)
y_pred = model.predict(features)
yaw_pred, pitch_pred, roll_pred = y_pred[0]
print('110_cropped')
print('Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('Yaw: {:.2f}°'.format(yaw_pred)) 





im = cv2.imread('./MLPTest/celeb_1.jpg', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)
features = compute_features(face_points)
features = std.transform(features)
y_pred = model.predict(features)
yaw_pred, pitch_pred, roll_pred = y_pred[0]
print('celeb_1')
print('Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('Yaw: {:.2f}°'.format(yaw_pred))


im = cv2.imread('./MLPTest/celeb_2.jpg', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)
features = compute_features(face_points)
features = std.transform(features)
y_pred = model.predict(features)
yaw_pred, pitch_pred, roll_pred = y_pred[0]
print('celeb_2')
print('Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('Yaw: {:.2f}°'.format(yaw_pred))


im = cv2.imread('./MLPTest/celeb_4.jpg', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)
features = compute_features(face_points)
features = std.transform(features)
y_pred = model.predict(features)
yaw_pred, pitch_pred, roll_pred = y_pred[0]
print('celeb_4')
print('Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('Yaw: {:.2f}°'.format(yaw_pred))
