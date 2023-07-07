Setup Instructions:

1.First configure your mobile hotspot by giving ID as "Sujan" and PASSWORD as "sujand072".
2.Connect your PC and Raspbeery PI to the same Wi-Fi.
3.Login to VNC Viewer by giving username as "pi" and password as "raspberry"
4.After login open project folder in desktop.
5.In that open face_datasets.py to ceate dataset. In that give the ID. and run.
6.After open the Motion.py file for motion detection and facerecognition.
7.For live streaming, in mobile find the IP adress of motion-eye which is connected to the same wi-fi.
8.Enter the IP address of that motion-eye in the browser or in the Motion eye application.


####################################################
For dataset creation
1. open face_datasets.py file
2. put face_id = number of users count like face_id = 1 for 1st user face_id = 2 for 2nd user.
3. save and run
4. photos are stored in the dataset folder.

###############################################
After dataset creation for taining the dataset

4. Run the training.py

##################################################
Face recognition

5. Run Motion.py file for motion detection and facerecognition.
