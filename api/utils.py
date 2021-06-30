import cv2
import numpy as np

def decode_image(data):
    nparr = np.fromstring(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)