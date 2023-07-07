import cv2
import os
import time
import RPi.GPIO as GPIO
import sys
import time
import random
import datetime

# Import numpy for matrices calculations
import numpy as np
import time
import datetime
import serial

##import atttachem
from twilio.rest import Client


from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(0.1)


##GPIO.setup(26, GPIO.IN, pull_up_down=GPIO.PUD_UP)  
# Import numpy for matrices calculations
import numpy as np
import time
import datetime
# Create Local Binary Patterns Histograms for face recognization
#recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('trainer/trainer.yml')
##recognizer.read('/home/pi/Desktop/face_recog_folder/Raspberry-Face-Recognition-master/trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
##cam = cv2.VideoCapture(0)

flag = []
count1=0
count2=0
count3=0
sample =0
lecture=0
mon=0
count=0
a=0


account_sid = "ACccc6b5c88322691206e98a423af622e2"
auth_token = "6aa66d8f3c4c3894fd7de9a1422e08cf"

client = Client(account_sid, auth_token)


while True:
 #for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
##  while True:
##            rawCapture.truncate(0)
##        if(GPIO.input(SW)==False):
##            print('Entry')
##            DOOR_OPEN()
##            time.sleep(2)
##            DOOR_CLOSE()
##            x = data.read(1)
##            x=x.decode('UTF-8','ignore')
##            print(x)
##            if x=='A':
##                print('Checking for multiple persons')
##            a=1
##        while a==1:
##            x = data.read(1)
##            x=x.decode('UTF-8','ignore')
##            print(x)
##            if x=='H':
##                print('Helmet detected')
##                ##client.api.account.messages.create(
####    to="+91-9902599273",
####    from_="+1 716-543-3315" ,  #+1 210-762-4855"
####    body=" Detected" )
##                a=0
##                
##            if x=='M':
##                print('Mask detected')
##                ##client.api.account.messages.create(
####    to="+91-9902599273",
####    from_="+1 716-543-3315" ,  #+1 210-762-4855"
####    body=" Detected" )
##                a=0

            now = datetime.datetime.now()

                # Read the video frame
            ret, im =cam.read()
            #im = frame.array

                # Convert the captured frame into grayscale
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            #rawCapture.truncate(0)

                # Get all face from the video frame
            faces = faceCascade.detectMultiScale(gray, 1.2,5)

                # For each face in faces
            for(x,y,w,h) in faces:

                    # Create rectangle around the face
                cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

                    # Recognize the face belongs to which ID
                Id,i = recognizer.predict(gray[y:y+h,x:x+w])
                    #id = int(os.path.split(imagePath)[-1].split(".")[1])
                    
                print(i)
            # Check the ID if exist
                if (len(faces)) > 1:
                    print('Found more than two persons in ATM')
                    lcd_byte(0x01, LCD_CMD)
                    lcd_string("Multiple Persons    ",LCD_LINE_1)
                    lcd_string("In ATM...   ",LCD_LINE_2)
                    cv2.putText(im,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)
                    cv2.imshow('frame', im)
                    cv2.imwrite('frame.png',im)
                   
                cv2.putText(im,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)
                cv2.imshow('frame', im)
                
            # To stop taking video, press 'q' for at least 100ms
            #cv2.putText(image_frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            cv2.imshow('frame', im)
    cam.release()

# Close all windows
    cv2.destroyAllWindows()

