from rest_framework.response import Response
from detect import detect_v8
from rest_framework.views import APIView
import base64
import numpy as np
import cv2


# DetectObjectView
class DetectProduct(APIView):
    def post(self, request, format=None):
        class_name = ""
        total_detection = ""
        status = 400
        try:
            image = request.data.get('image')
            image_base64_decode = base64.b64decode(image)
            im_arr = np.frombuffer(image_base64_decode, dtype=np.uint8)
            image_np = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR) 
            class_name, total_detection = detect_v8(image_np)
            status = 200

        except Exception as e:
            status = 400
        return Response({'status':status, 'class_name' : class_name, "Object": total_detection})