import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainer/trainer.yml')

cascadePath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)

flag = []
count1=0
count2=0
count3=0
sample =0
lecture=0
mon=0
count=0

while True:

        ret, im =cam.read()

        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.2,5)

        for(x,y,w,h) in faces:

            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
           
            Id,i = recognizer.predict(gray[y:y+h,x:x+w])     
            print(i)
            Id1=''
            if i < 60:
                sample= sample+1
                if Id == 1 :
                    count1=1
                    Id1 = "Dilip"
                    print("Dilip")
                    lecture=1
                    sample=0
                if Id == 2 :
                    count1=1
                    Id1 = "Sujan"
                    print("Sujan")
                    lecture=1
                    sample=0


##                if Id == 2 :
##                    #flag[1]=1
##                    count1=1
##                    Id = ""
##                    print("")
##                    lecture=1
##                    sample=0
##                    break                       
            else:
                count=count+1

                if count > 5:
                    count=0
                    print(Id)                
                    Id1 = "unknown"                  
                    print('UNKNOWN PERSON') 
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, str(Id1), (x,y-40), font, 2, (255,255,255), 3)

        cv2.imshow('im',im)
        
        if cv2.waitKey(20) & 0xFF == ord('q'): 
            break
           
cam.release()


cv2.destroyAllWindows()
