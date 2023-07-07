# USAGE
# python3 motion_detector.py --yolo yolo-coco


# import the necessary packages
import numpy as np
import argparse
import datetime
import imutils
import time
import cv2
import os
import platform
import pyttsx3
from gtts import gTTS

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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size") #500


####################################
# code from yolo.py START 1/2
####################################

ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")

####################################
# code from yolo.py END 1/2
####################################


args = vars(ap.parse_args())	#argument

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)	#delay for processing - opening webcam

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])





def test(cnts):# loop over the frames of the video
	# loop over the contours
	for c in cnts:
		print('entering the looooooooooop')

		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue

		# compute the bounding box for the contour		
		count += 1
		(x, y, w, h) = cv2.boundingRect(c)
		
		# draw the bounding box for the contour on the frame
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)  #2

		# update the text
		status_text = "Occupied"

	return status_text







def contour_func():

	print('entered : 1')
	
	status_text = "Unoccupied"

	# initialize the first frame in the video stream
	firstFrame = None

	(grabbed, frame) = camera.read()
	print('entered : 2')


	# if the frame could not be grabbed, then we have reached the end of the video
	if not grabbed:
		return


	print('entered : 3')

	cv2.imwrite("test.jpg", frame)

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		# continue

	print('entered : 4')

	# compute the absolute difference between the current frame and first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]  #25

	# dilate the thresholded image to fill in holes, then find contours on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

	count = 0

	print('entered : 5')


	print(count)

	# if count>3:	#6
		# cv2.imshow("activity detected",frame)

	status_text = test(cnts)
	if status_text == 'Occupied':
		status_text = 'Occupied'


	# draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(status_text), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  #0.5
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	# cv2.imshow("Thresh", thresh)
	# cv2.imshow("Frame Delta", frameDelta)

	return



# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# loop over the frames of the video
while True:
	global status_text 
	# place 1
	print('place1')
	contour_func()


	####################################
	# code from yolo.py START 2/2
	####################################
	
	print("[INFO] loading YOLO from disk...")
	
	
	# load our input image and grab its spatial dimensions
	image = cv2.imread("test.jpg")
	(H, W) = image.shape[:2]

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# place 2
	contour_func()


	# show timing information on YOLO
	print("[INFO] YOLO took {:.1f} seconds\n".format(end - start))

	mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448
	mem_gib = mem_bytes/(1024.**3)  # e.g. 1.9
	print("[INFO] You have only {:.1f}GB of RAM".format(mem_gib))
	print('Increase your RAM size for better performance\n')

	# processor
	print("[INFO]  Processors: ")
	with open("/proc/cpuinfo", "r")  as f:
    		info = f.readlines()

	cpuinfo = [x.strip().split(":")[1] for x in info if "model name"  in x]
	for index, item in enumerate(cpuinfo):
		print("    " + str(index) + ": " + item)
		break
	print('Use GPU instead of CPU for better performance\n')

	print('-----------------------------------------------------------------------------------')

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# place 3
	contour_func()
	

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)


	# place 4
	contour_func()

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])



	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			# to be changed to knife instead of person
			if LABELS[classIDs[i]] == 'person':
				print("Danger... It's a knife ! \n")
				# tts = gTTS(text='Danger there is a knife', lang='en')
				# tts.save("sound.mp3")
				os.system("mpg321 sound.mp3")

	# show the output image
	# cv2.imshow("Image@@", image)
	# cv2.waitKey(0)
	
	
	# place 5
	contour_func()

	# show the output image
	cv2.imshow("Object detection", image)


	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

