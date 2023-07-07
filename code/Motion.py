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

## initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
## allow the camera to warmup
time.sleep(0.1)



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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])
# initialize the first frame in the video stream
firstFrame = None
count=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
count9=0
count10=0
count11=0
count12=0
# loop over the frames of the video
#while True:
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the current frame and initialize the occupied/unoccupied
	# text
    #frame = vs.read()
    frame = frame.array
    im=frame.copy()
    frame = frame if args.get("video", None) is None else frame[1]
    text = "No Motion Detected"
    rawCapture.truncate(0)
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
        #print('Area {}'.format(cv2.contourArea(c)))
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Motion Detected"
        if cv2.contourArea(c) > 1000:
            print('Loading...')
            GPIO.output(RL,True)
            time.sleep(5)
        else:
            GPIO.output(RL,False)
            time.sleep(1)

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
                Id,i = recognizer.predict(gray[y:y+h,x:x+w])     
                print(i)
                Id1=''
                if i < 60:
                        """
                        if Id == 1: 
                            Id1 = "Mr. Sujan D"
                            print(Id1)
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
                        """
                        if Id == 2: 
                            Id1 = "Supreeth"
                            print("Supreeth")
                            count2=count2+1
                            if count2 > 5:
                                count2=0
                                
                                cv2.imwrite('/home/pi/Desktop/Project/frame.png',frame)
                                bot = telepot.Bot('5489502707:AAHFTiATFb4kxre7TOJlGver9_1GPjPYmNU')
                                bot.sendMessage('5152706156', str('Known Person Detected'+ ' '+Id1))
                                bot.sendPhoto('5152706156',photo=open('/home/pi/Desktop/Project/frame.png','rb'))
                                
                                message = client.messages.create(  
                                  from_='+17124017907', 
                                  body='Known Person {}'.format(Id1),      
                                  to='+919380641474'
                                  )

                                
                        if Id == 3: 
                            Id1 = "Dilip"
                            print("Dilip")
                            count3=count3+1
                            if count3 > 5:
                                count3=0
                                
                                cv2.imwrite('/home/pi/Desktop/Project/frame.png',frame)
                                bot = telepot.Bot('5489502707:AAHFTiATFb4kxre7TOJlGver9_1GPjPYmNU')
                                bot.sendMessage('5152706156', str('Known Person Detected'+' '+ Id1))
                                bot.sendPhoto('5152706156',photo=open('/home/pi/Desktop/Project/frame.png','rb'))
                                
                                message = client.messages.create(  
                                  from_='+17124017907', 
                                  body='Known Person {}'.format(Id1),      
                                  to='+919380641474'
                                  )
                                
                        if Id == 4: 
                            Id1 = "T Dhruvkumar"
                            print(Id1)
                            count4=count4+1
                            if count4 > 5:
                                count4=0
                                
                                cv2.imwrite('/home/pi/Desktop/Project/frame.png',frame)
                                bot = telepot.Bot('5489502707:AAHFTiATFb4kxre7TOJlGver9_1GPjPYmNU')
                                bot.sendMessage('5152706156', str('Known Person Detected'+' '+ Id1))
                                bot.sendPhoto('5152706156',photo=open('/home/pi/Desktop/Project/frame.png','rb'))
                                
                                message = client.messages.create(  
                                  from_='+17124017907', 
                                  body='Known Person {}'.format(Id1),      
                                  to='+919380641474'
                                  )

                                
                        if Id == 5: 
                            Id1 = "Harshalatha Y"
                            print(Id1)
                            count5=count5+1
                            if count5 > 5:
                                count5=0
                                
                                cv2.imwrite('/home/pi/Desktop/Project/frame.png',frame)
                                bot = telepot.Bot('5489502707:AAHFTiATFb4kxre7TOJlGver9_1GPjPYmNU')
                                bot.sendMessage('5152706156', str('Known Person Detected'+' '+ Id1))
                                bot.sendPhoto('5152706156',photo=open('/home/pi/Desktop/Project/frame.png','rb'))
                                
                                message = client.messages.create(  
                                  from_='+17124017907', 
                                  body='Known Person {}'.format(Id1),      
                                  to='+919380641474'
                                  )

                                
                        if Id == 6: 
                            Id1 = "S Raveesh"
                            print(Id1)
                            count6=count6+1
                            if count6 > 5:
                                count6=0
                                
                                cv2.imwrite('/home/pi/Desktop/Project/frame.png',frame)
                                bot = telepot.Bot('5489502707:AAHFTiATFb4kxre7TOJlGver9_1GPjPYmNU')
                                bot.sendMessage('5152706156', str('Known Person Detected'+' '+ Id1))
                                bot.sendPhoto('5152706156',photo=open('/home/pi/Desktop/Project/frame.png','rb'))
                                
                                message = client.messages.create(  
                                  from_='+17124017907', 
                                  body='Known Person {}'.format(Id1),      
                                  to='+919380641474'
                                  )

                                
                        if Id == 7: 
                            Id1 = "Seema B Hegde"
                            print(Id1)
                            count7=count7+1
                            if count7 > 5:
                                count7=0
                                
                                cv2.imwrite('/home/pi/Desktop/Project/frame.png',frame)
                                bot = telepot.Bot('5489502707:AAHFTiATFb4kxre7TOJlGver9_1GPjPYmNU')
                                bot.sendMessage('5152706156', str('Known Person Detected'+ ' '+Id1))
                                bot.sendPhoto('5152706156',photo=open('/home/pi/Desktop/Project/frame.png','rb'))
                                
                                message = client.messages.create(  
                                  from_='+17124017907', 
                                  body='Known Person {}'.format(Id1),      
                                  to='+919380641474'
                                  )

                                
                        if Id == 8: 
                            Id1 = "T C Mahalingesh"
                            print(Id1)
                            count8=count8+1
                            if count8 > 5:
                                count8=0
                                
                                cv2.imwrite('/home/pi/Desktop/Project/frame.png',frame)
                                bot = telepot.Bot('5489502707:AAHFTiATFb4kxre7TOJlGver9_1GPjPYmNU')
                                bot.sendMessage('5152706156', str('Known Person Detected'+ ' '+Id1))
                                bot.sendPhoto('5152706156',photo=open('/home/pi/Desktop/Project/frame.png','rb'))
                                
                                message = client.messages.create(  
                                  from_='+17124017907', 
                                  body='Known Person {}'.format(Id1),      
                                  to='+919380641474'
                                  )
                                  
                        if Id == 10: 
                            Id1 = "V M Aparanji"
                            print(Id1)

                            count10=count10+1
                            if count10 > 5:
                                count10=0
                                
                                cv2.imwrite('/home/pi/Desktop/Project/frame.png',frame)
                                bot = telepot.Bot('5489502707:AAHFTiATFb4kxre7TOJlGver9_1GPjPYmNU')
                                bot.sendMessage('5152706156', str('Known Person Detected'+ ' '+Id1))
                                bot.sendPhoto('5152706156',photo=open('/home/pi/Desktop/Project/frame.png','rb'))
                                
                                message = client.messages.create(  
                                  from_='+17124017907', 
                                  body='Known Person {}'.format(Id1),      
                                  to='+919380641474'
                                  )

                        if Id == 11: 
                            Id1 = "Apoorva Y S"
                            print(Id1)

                            count11=count11+1
                            if count11 > 5:
                                count11=0
                                
                                cv2.imwrite('/home/pi/Desktop/Project/frame.png',frame)
                                bot = telepot.Bot('5489502707:AAHFTiATFb4kxre7TOJlGver9_1GPjPYmNU')
                                bot.sendMessage('5152706156', str('Known Person Detected'+ ' '+Id1))
                                bot.sendPhoto('5152706156',photo=open('/home/pi/Desktop/Project/frame.png','rb'))
                                
                                message = client.messages.create(  
                                  from_='+17124017907', 
                                  body='Known Person {}'.format(Id1),      
                                  to='+919380641474'
                                  )

                        if Id == 12: 
                            Id1 = "B N Shashikala"
                            print(Id1)

                            count1=count1+1
                            if count1 > 5:
                                count1=0
                                
                                cv2.imwrite('/home/pi/Desktop/Project/frame.png',frame)
                                bot = telepot.Bot('5489502707:AAHFTiATFb4kxre7TOJlGver9_1GPjPYmNU')
                                bot.sendMessage('5152706156', str('Known Person Detected'+' '+ Id1))
                                bot.sendPhoto('5152706156',photo=open('/home/pi/Desktop/Project/frame.png','rb'))
                                
                                message = client.messages.create(  
                                  from_='+17124017907', 
                                  body='Known Person {}'.format(Id1),      
                                  to='+919380641474'
                                  )

                        if Id == 18: 
                            Id1 = "Sujan D"
                            print("Sujan D")

                            count12=count12+1
                            if count12 > 5:
                                count12=0
                                
                                cv2.imwrite('/home/pi/Desktop/Project/frame.png',frame)
                                bot = telepot.Bot('5489502707:AAHFTiATFb4kxre7TOJlGver9_1GPjPYmNU')
                                bot.sendMessage('5152706156', str('Known Person Detected'+' '+Id1))
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
