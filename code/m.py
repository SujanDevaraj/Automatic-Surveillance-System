# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2

import numpy as np
import time
import datetime

from PIL import Image

import time
import RPi.GPIO as GPIO
import datetime

import random
import datetime
import telepot


from picamera.array import PiRGBArray
from picamera import PiCamera
##import attachem

## initialize the camera and grab a reference to the raw camera capt
#camera = PiCamera()
#time.sleep(0.1)


from twilio.rest import Client
account_sid = 'AC0ab507cfd761e663d18dcfaff191a339' 
auth_token = '8e3c7e26a1bf9677f28e5b6eb0a4f36d' 

client = Client(account_sid, auth_token)


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
RL=23
GPIO.setup(RL,GPIO.OUT)
GPIO.output(RL,False)


# Create Local Binary Patterns Histograms for face recognization
#recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('/home/pi/Desktop/Project/trainer/trainer.yml')


# Load prebuilt model for Frontal Face
cascadePath = "/home/pi/Desktop/Project/haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX


count=0
count1=0
# loop over the frames of the video
#while True:
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)    
   
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
                Id,i = recognizer.predict(gray[y:y+h,x:x+w])     
                print(i)
                Id1=''
                if i < 60:
                        if Id == 1: 
                            Id1 = "Sujan"
                            print("Sujan")

                        if Id == 18: 
                            Id1 = "Dilip"
                            print("Dilip")

                            count1=count1+1
                            if count1 > 5:
                                count1=0
                                cv2.imwrite('/home/pi/Desktop/Project/frame.png',frame)
                                bot = telepot.Bot('5489502707:AAHFTiATFb4kxre7TOJlGver9_1GPjPYmNU')
                                bot.sendMessage('5152706156', str('Known Person Detected'+ Id1))
                                bot.sendPhoto('5152706156',photo=open('/home/pi/Desktop/Project/frame.png','rb'))
                                message = client.messages.create(  
                                  from_='+17124017907', 
                                  body='Known Person {}'.format(Id1),      
                                  to='+919380641474'
                                  ) 
                            
                else:
                        count=count+1

                        if count > 5:
                            count=0
                            print(Id)                
                            Id1 = "unknown"                  
                            print('UNKNOWN PERSON') 
                            cv2.imwrite("/home/pi/Desktop/Project/dataset1/User." + str(Id1) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                            cv2.imwrite('frame.png',frame)
                            bot = telepot.Bot('5489502707:AAHFTiATFb4kxre7TOJlGver9_1GPjPYmNU')
                            bot.sendMessage('5152706156', str('Unknown Person Detected'))
                            bot.sendPhoto('5152706156',photo=open('/home/pi/Desktop/Project/frame.png','rb'))
                            message = client.messages.create(  
                              from_='+17124017907', 
                              body='UnKnown Person detected in survillance area.',      
                              to='+919380641474' 
                          ) 
                            
                cv2.rectangle(frame, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
                cv2.putText(frame, str(Id1), (x,y-40), font, 2, (255,255,255), 3)
    cv2.imshow('im',frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
