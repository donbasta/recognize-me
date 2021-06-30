import cv2
import numpy as np
from .constants import HAARCASCADE_FRONTALFACE_ALT, HAARCASCADE_FRONTALFACE_DEFAULT

FACE_DETECTION_MODEL = [HAARCASCADE_FRONTALFACE_ALT, HAARCASCADE_FRONTALFACE_DEFAULT]

class FaceNotFoundException(Exception):
    def __init__(self, msg):
        pass
    def __str__(self):
        return "Face not found in the image"

class FrameProcessor(object):
    def __init__(self, detection_model=HAARCASCADE_FRONTALFACE_ALT):
        self.face_detector_model = detection_model

    def crop_face(self, imgarray, section, margin=20, size=224):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def extract_faces(self, img_array):
        face_cascade = cv2.CascadeClassifier(self.face_detector_model)

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=10,
            minSize=(64, 64)
        )

        face = None
        if len(faces) > 1:
            face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))
        elif len(faces) == 1:
            face = faces[0]

        face_img = None
        cropped = None
        if face is not None:
            face_img, cropped = self.crop_face(img_array, face, margin=40, size=224)
        
        return face_img, cropped