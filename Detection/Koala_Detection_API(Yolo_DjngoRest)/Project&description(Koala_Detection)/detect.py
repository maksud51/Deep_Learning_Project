import os
import uuid
import cv2
from collections import Counter
from ultralytics import YOLO

from django.conf import settings
ROOT = settings.MEDIA_ROOT 
model = YOLO(' H:/project/project/best.pt')   
product_list = ["kola"]## change this with your class name

def detect_v8(image):
    results = model.predict(source=image)  
    tensor_list = results[0].boxes.data
    detection = tensor_list.tolist()
    total_detection = len(detection)
    filename = f"{uuid.uuid4().hex}.png"
    savepath = os.path.join(ROOT, filename)
    for det in detection:
        x,y,w,h,confidence,cls = [int(d) for d in det]
        class_name = str(product_list[int(cls)])
        image = cv2.rectangle(image,(x,y),(w,h),(255,0,0), 2)
    cv2.imwrite(savepath, image)
    return class_name, total_detection
