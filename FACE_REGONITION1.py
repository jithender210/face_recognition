import cv2
import numpy as np

def draw_border(img,face_cascode,colour,scale,minneighbours,clf):
    coordinates =[]
    gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascode.detectMultiScale(gray,scale,minneighbours)
    
    for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),colour,2)
            id,confidence=clf.predict(gray[y:y+h,x:x+w])
            #cv2.putText(img,f"id: {id}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,colour,2)
            if confidence < 60:
                cv2.putText(img,"hello Jithender",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,colour,2)
            print(f"ID: {id}, Confidence: {confidence}")
            coordinates = [x,y,w,h]
    return coordinates
 
def generate_lbp(image,user_id,img_id):
     cv2.imwrite("samples/user-"+str(user_id)+"-"+"img_id"+str(img_id)+".jpg",image)

def recognize(img,clf,face_cascode):
     colour={"blue":(255,0,0),"green":(0,255,0),"red":(0,0,255),"yellow":(0,255,255)}
     coord= draw_border(img,face_cascode,colour["blue"],1.1,5,clf)
     return img
    
def find_face(img,face_cascode,eye_cascode,node_cascode,mouth_cascode,img_id):
    colour={"blue":(255,0,0),"green":(0,255,0),"red":(0,0,255),"yellow":(0,255,255)}
    coord= draw_border(img,face_cascode,colour["blue"],1.1,5)
    user_id=1
    if len(coord) == 4:
         roi_img=img[coord[1]:coord[1]+coord[3],coord[0]:coord[0]+coord[2]]
         generate_lbp(roi_img,user_id,img_id)
        #  coord=draw_borderce_detech(roi_img,eye_cascode,colour["red"],1.1,14)
        #  coord=draw_border(roi_img,node_cascode,colour["yellow"],1.1,10)
        #  coord=draw_border(roi_img,mouth_cascode,colour["green"],1.1,20)
        
    
    return img

cap=cv2.VideoCapture(0)
cv2.namedWindow("webcam",cv2.WINDOW_NORMAL)
cv2.resizeWindow("webcam",640,480)
face_cascode =cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascode =cv2.CascadeClassifier('haarcascade_eye.xml')
node_cascode =cv2.CascadeClassifier('Nariz.xml')
mouth_cascode =cv2.CascadeClassifier('Mouth.xml')
lbp_cascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")
clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("trainer.yml")
img_id=0
while cap.isOpened():
    r,frame=cap.read()
    if r == True:
        frame=cv2.flip(frame,1)
        
        #img = find_face(frame,face_cascode,eye_cascode,node_cascode,mouth_cascode,img_id)
         
        img=recognize(frame,clf,face_cascode)
        img_id+=1
        cv2.imshow("webcam",img)
        if cv2.waitKey(1)& 0xff == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()