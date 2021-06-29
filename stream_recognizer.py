import cv2
import os
import glob
import pickle
import numpy as np

from constants import *

from keras.engine import Model
from keras import models
from keras import layers
from keras.layers import Input
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from scipy.spatial.distance import cosine, euclidean

def load_embedding(filename):
    saved_stuff = open(filename, "rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff

class FaceIdentifyStream(object):
    """
    Class for real time face recognition (video stream input)
    """
    FACE_DETECTOR_PATH = [HAARCASCADE_FRONTALFACE_ALT, HAARCASCADE_FRONTALFACE_DEFAULT]

    def __new__(cls, feature_embedding_file=None):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceIdentifyStream, cls).__new__(cls)
        return cls.instance
    
    def __init__(self, feature_embedding_file=None):
        self.face_size = FACE_DIM
        self.precompute_face_embedding_map = load_embedding(feature_embedding_file)
        print("Loading VGG Face model...")
        self.model = VGGFace(model='senet50', 
                             include_top=False, 
                             input_shape=(224, 224, 3), 
                             pooling='avg')
        print("Loading model done...")
    
    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=20, size=FACE_DIM):
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
        margin = int(min(w, h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resize_img = np.array(resized_img)
        return resize_img, (x_a, y_a, x_b - x_a, y_b - y_a)
    
    def identify_face(self, features, threshold=200, metric="euclidean"):
        min_dist_val = NUM_MAX
        min_dist_idx = -1
        for idx, person in enumerate(self.precompute_face_embedding_map):
            person_features = person.get("features")
            if metric == "euclidean":
                distance = euclidean(person_features, features)
            elif metric == "cosine":
                distance = cosine(person_features, features)
            name = person.get("name")
            print(f"name: {name},  distance = {distance}")
            if distance < min_dist_val:
                min_dist_val = distance
                min_dist_idx = idx
        print(f"Min_dist_val: {min_dist_val}")
        if min_dist_val <= threshold:
            return self.precompute_face_embedding_map[min_dist_idx].get("name")
        else:
            return "Not recognized from database"
    
    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.FACE_DETECTOR_PATH[0])

        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            if not video_capture.isOpened():
                sleep(5)
            # Capture frame by frame from the video
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(64, 64)
            )

            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
            for i, face in enumerate(faces):
                face_img, cropped = self.crop_face(frame, face, margin=10, size=self.face_size)
                (x, y, w, h) = cropped
                cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 200, 0), 2)
                face_imgs[i, :, :, :] = face_img
            if len(face_imgs) > 0:
                feature_faces = self.model.predict(face_imgs)
                predicted_names = [self.identify_face(feature_face) for feature_face in feature_faces]
            
            for i, face in enumerate(faces):
                label = f"{predicted_names[i]}"
                self.draw_label(frame, (face[0], face[1]), label)
            
            cv2.imshow('Keras Faces', frame)
            if cv2.waitKey(5) == ESC_KEY: 
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        return

def main():
    face = FaceIdentifyStream(feature_embedding_file="./data/embeddings/tes2.pickle")
    face.detect_face()

if __name__ == "__main__":
    main()