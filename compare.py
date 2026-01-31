import numpy as np 
import os ,cv2
from PIL import Image


def transform_img(data):
    path=[os.path.join(data,f) for f in os.listdir(data) ]
    faces=[]
    ids=[]
    for image in path:
        img=Image.open(image).convert("L")
        imgnp=np.array(img,"uint8")
        faces.append(imgnp)
        ids.append(int(os.path.split(image)[1].split("-")[1].split(".")[0]))
        
    ids=np.array(ids)
    clf=cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("eye_trainer.yml")


transform_img("samples")