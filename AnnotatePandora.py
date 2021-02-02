from tkinter import *
#from tkinter import *
# import ttk
from tkinter import ttk
#from PIL import Image, ImageTk
import imutils
import dlib
import cv2
import numpy as np
import math
from PIL import ImageTk, Image
import os
#csv
import csv


root = Tk()  
root.title("Pandora GT Correction")
csv_file = ('./rpy_gt15.csv')
pitch = DoubleVar()
yaw = DoubleVar()
roll = DoubleVar()
strng = StringVar()
ind = IntVar()

#Display adjustable bounding box for the euler angles as i/p
def getvalue(allPng,panel):
    strng = allPng[ind]
    print(strng)
    print(pitch.get())
    print(yaw.get())
    print(roll.get())
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    img = cv2.imread(strng, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    size = image.shape
    rects = detector(image, 1)
    for rect in rects:
        dlib_points = predictor(image, rect)
        image_points = np.array([(dlib_points.part(17).x,dlib_points.part(17).y),(dlib_points.part(21).x,dlib_points.part(21).y),(dlib_points.part(22).x,dlib_points.part(22).y),(dlib_points.part(26).x,dlib_points.part(26).y),(dlib_points.part(36).x,dlib_points.part(36).y),(dlib_points.part(39).x,dlib_points.part(39).y),(dlib_points.part(42).x,dlib_points.part(42).y),(dlib_points.part(45).x,dlib_points.part(45).y),(dlib_points.part(31).x,dlib_points.part(31).y),(dlib_points.part(35).x,dlib_points.part(35).y),(dlib_points.part(48).x,dlib_points.part(48).y),(dlib_points.part(54).x,dlib_points.part(54).y),(dlib_points.part(57).x,dlib_points.part(57).y),(dlib_points.part(8).x,dlib_points.part(8).y)], dtype="double")  
        model_points = np.array([(6.825897, 6.760612, 4.402142),(1.330353, 7.122144, 6.903745),(-1.330353, 7.122144, 6.903745),(-6.825897, 6.760612, 4.402142),(5.311432, 5.485328, 3.987654),(1.789930, 5.393625, 4.413414),(-1.789930, 5.393625, 4.413414),(-5.311432, 5.485328, 3.987654),(2.005628, 1.409845, 6.165652),(-2.005628, 1.409845, 6.165652),(2.774015, -2.080775, 5.048531),(-2.774015, -2.080775, 5.048531),(0.000000, -3.116408, 6.097667),(0.000000, -7.415691, 4.070434)])
        reprojectsrc = np.array([(10.0, 10.0, 10.0),(10.0, 10.0, -10.0),(10.0, -10.0, -10.0),(10.0, -10.0, 10.0),(-10.0, 10.0, 10.0),(-10.0, 10.0, -10.0),(-10.0, -10.0, -10.0),(-10.0, -10.0, 10.0)])
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
        dist_coeffs = np.zeros((4,1))
  
        #Input gt pyr values in the format pitch,yaw,roll in radians
        theta = np.array([pitch.get(),yaw.get(),roll.get()],dtype='float')
        #Convert to degrees
        theta[0] = theta[0]*3.14/180
        theta[1] = theta[1]*3.14/180
        theta[2] = theta[2]*3.14/180
        print(theta[0])
        #Rotation matrix around x-axis
        R_x = np.array([[1,0,0],[0,math.cos(theta[0]), -math.sin(theta[0]) ],[0, math.sin(theta[0]), math.cos(theta[0])]])
        #Rotation matrix around y-axis
        R_y = np.array([[math.cos(theta[1]),0,math.sin(theta[1])],[0,1,0],[-math.sin(theta[1]),0,math.cos(theta[1])]])
        #Rotation matrix around z-axis
        R_z = np.array([[math.cos(theta[2]),-math.sin(theta[2]),0],[math.sin(theta[2]),math.cos(theta[2]),0],[0,0,1]])
        #Rotation matrix around x,y and z axis
        rotationMat = np.dot(R_z,np.dot(R_y,R_x))
        #solvePnP() used only for getting translation_vector
        #Dont use rotational vector
        (success, rotation_vector_orig, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        
        (rotation_vector,jac)=cv2.Rodrigues(rotationMat)
        (reprojectdst,jacobian) = cv2.projectPoints(reprojectsrc,rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        #Correct the translation_vector
        corrX = dlib_points.part(26).x - reprojectdst[5][0][0]
        corrY = dlib_points.part(25).y - reprojectdst[5][0][1]
        #Draw bounding box lines
        cv2.line(image,(int(reprojectdst[0][0][0]+corrX), int(reprojectdst[0][0][1]+corrY)),(int(reprojectdst[1][0][0]+corrX), int(reprojectdst[1][0][1]+corrY)),(0,0,255),2)
        cv2.line(image,(int(reprojectdst[1][0][0]+corrX), int(reprojectdst[1][0][1]+corrY)),(int(reprojectdst[2][0][0]+corrX), int(reprojectdst[2][0][1]+corrY)),(0,255,0),2)
        cv2.line(image,(int(reprojectdst[2][0][0]+corrX), int(reprojectdst[2][0][1]+corrY)),(int(reprojectdst[3][0][0]+corrX), int(reprojectdst[3][0][1]+corrY)),(0,0,255),2)
        cv2.line(image,(int(reprojectdst[3][0][0]+corrX), int(reprojectdst[3][0][1]+corrY)),(int(reprojectdst[0][0][0]+corrX), int(reprojectdst[0][0][1]+corrY)),(0,0,255),2)
        cv2.line(image,(int(reprojectdst[4][0][0]+corrX), int(reprojectdst[4][0][1]+corrY)),(int(reprojectdst[5][0][0]+corrX), int(reprojectdst[5][0][1]+corrY)),(0,0,255),2)
        cv2.line(image,(int(reprojectdst[5][0][0]+corrX), int(reprojectdst[5][0][1]+corrY)),(int(reprojectdst[6][0][0]+corrX), int(reprojectdst[6][0][1]+corrY)),(0,255,0),2)
        cv2.line(image,(int(reprojectdst[6][0][0]+corrX), int(reprojectdst[6][0][1]+corrY)),(int(reprojectdst[7][0][0]+corrX), int(reprojectdst[7][0][1]+corrY)),(0,0,255),2)
        cv2.line(image,(int(reprojectdst[7][0][0]+corrX), int(reprojectdst[7][0][1]+corrY)),(int(reprojectdst[4][0][0]+corrX), int(reprojectdst[4][0][1]+corrY)),(0,0,255),2)
        cv2.line(image,(int(reprojectdst[0][0][0]+corrX), int(reprojectdst[0][0][1]+corrY)),(int(reprojectdst[4][0][0]+corrX), int(reprojectdst[4][0][1]+corrY)),(0,0,255),2)
        cv2.line(image,(int(reprojectdst[1][0][0]+corrX), int(reprojectdst[1][0][1]+corrY)),(int(reprojectdst[5][0][0]+corrX), int(reprojectdst[5][0][1]+corrY)),(0,255,0),2)
        cv2.line(image,(int(reprojectdst[2][0][0]+corrX), int(reprojectdst[2][0][1]+corrY)),(int(reprojectdst[6][0][0]+corrX), int(reprojectdst[6][0][1]+corrY)),(0,255,0),2)
        cv2.line(image,(int(reprojectdst[3][0][0]+corrX), int(reprojectdst[3][0][1]+corrY)),(int(reprojectdst[7][0][0]+corrX), int(reprojectdst[7][0][1]+corrY)),(0,0,255),2)
        img = ImageTk.PhotoImage(image = Image.fromarray(image))
        panel.configure(image=img)
        panel.image = img

#Method to get next image from dataset..use ind variable as global
def updateNxt(allPng,panel,label0):
    global ind
    ind = ind + 1
    if ind!=len(allPng)-1:
        strng = allPng[ind]
        label0.config(text = strng)
        print(strng)
        print(ind)
        img = ImageTk.PhotoImage(Image.open(strng))
        panel.configure(image=img)
        panel.image = img
        getvalue(allPng, panel)
    else:
        ind = ind -1

def updateCombo(allPng, panel, imgSel):
    global ind
    ind = allPng.index(imgSel)
    strng = allPng[ind]
    label0.config(text = strng)
    print(imgSel)
    print(strng)
    print(ind)

    img = ImageTk.PhotoImage(Image.open(strng))
    panel.configure(image=img)
    panel.image = img
    getvalue(allPng, panel)


#Method to get previous image from dataset..use ind variable as global    
def updatePrev(allPng,panel,label0):
    global ind
    ind = ind - 1
    if ind!=-1:
        strng = allPng[ind]
        label0.config(text = strng)
        print(strng)
        print(ind)
        img = ImageTk.PhotoImage(Image.open(strng))
        panel.configure(image=img)
        panel.image = img
        getvalue(allPng, panel)
    else:
        ind = ind +1    

#Save off new GT value for image
def saveRPY(allPNG, panel,csvOpen):
    f = allPNG[ind]
    roll = scale3.get()
    pitch = scale1.get()
    yaw = scale2.get()
    row = [f, roll, pitch, yaw]
#    with open(csv_file,mode='wb') as csvfile:
#            csvwriter = csv.writer(csvfile)
#            csvwriter.writerow(row)
    c = csv.writer(csvOpen,dialect = 'excel')
    c.writerow(row)
    updateNxt(allPng, panel, label0)

#Get list of all png files
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

dirName = "./preprocessed_pandora/"
allPng = getListOfPNGFiles(dirName)
allPng.sort()
#Index Variable used globally


ind = 0
strng = allPng[ind]
label0 = Label(root,text=strng, fg="dark green")
#label0.pack()
   
n = StringVar()
imageChosen = ttk.Combobox(root,width=60,textvariable=n)
imageChosen['values']=allPng
imageChosen.current()
#imageChosen.pack()
comboButton = Button(root,command=lambda:updateCombo(allPng,panel,imageChosen.get()),text = 'Skip to Image',fg ='red') 
#comboButton.pack()
img = ImageTk.PhotoImage(Image.open(strng))
panel = Label(root, image = img)
panel.image = img # keep a reference
#panel.pack()

   
label0.grid(row = 0, column = 1, pady = 2)
imageChosen.grid(row = 1, column = 1, pady = 2) 
comboButton.grid(row = 2, column = 1, pady = 2) 
panel.grid(row = 3, column = 1, pady = 2,rowspan=6)

csvOpen = open(csv_file,mode='wb')

label1 = Label(root,text='Pitch', fg="dark green")
#label1.pack()
#Slider
scale1 = Scale(root,variable = pitch,from_=-60,to=60,resolution=0.2,orient=HORIZONTAL,sliderlength=25,length=150)
#scale1.pack()
label2 = Label(root,text='Yaw', fg="dark green")
#label2.pack()
scale2 = Scale(root,variable = yaw,from_=-60,to=60,resolution=0.2,orient=HORIZONTAL,sliderlength=25,length=150)
#scale2.pack()
label3 = Label(root,text='Roll', fg="dark green")
#label3.pack()
scale3 = Scale(root,variable = roll,from_=-60,to=60,resolution=0.2,orient=HORIZONTAL,sliderlength=25,length=150)
#scale3.pack()
bluebutton = Button(root,command=lambda:updateNxt(allPng,panel,label0),text = 'Next',fg ='blue') 
#bluebutton.pack()
redbutton = Button(root,command=lambda:getvalue(allPng,panel),text = 'Visualize',fg ='red') 
#redbutton.pack()
greenbutton = Button(root,command=lambda:updatePrev(allPng,panel,label0),text = 'Previous',fg ='green') 
#greenbutton.pack()  
savebutton = Button(root, command=lambda:saveRPY(allPng,panel,csvOpen),text = 'Save RPY', fg = 'black')
#savebutton.pack()

label1.grid(row = 3, column = 0, pady = 2)
label2.grid(row = 5, column = 0, pady = 2) 
label3.grid(row = 7, column = 0, pady = 2) 
scale1.grid(row = 4, column = 0, pady = 2)
scale2.grid(row = 6, column = 0, pady = 2)
scale3.grid(row = 8, column = 0, pady = 2)
bluebutton.grid(row = 3, column = 2, pady = 2)
redbutton.grid(row = 4, column = 2, pady = 2)
greenbutton.grid(row = 5, column = 2, pady = 2)
savebutton.grid(row = 6, column = 2, pady = 2)

root.mainloop()