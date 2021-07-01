import pafy
import cv2
import os
import glob
import pickle
import numpy as np
import time

from preprocessor.frame_processor import FrameProcessor

from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from scipy.spatial.distance import cosine, euclidean
from imutils.video import FPS

def load_embedding(filename):
    saved_stuff = open(filename, "rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff

class YoutubeStream(object):
    """
    Class for real time face recognition (video stream input)
    """

    def __new__(cls, embedding=None):
        if not hasattr(cls, 'instance'):
            cls.instance = super(YoutubeStream, cls).__new__(cls)
        return cls.instance
    
    def __init__(self, embedding=None):
        self.precompute_face_embedding_map = load_embedding(embedding)
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.processor = FrameProcessor()
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
    
    def identify_face(self, features, threshold=200, metric="euclidean"):
        min_dist_val = 1000000000
        min_dist_idx = -1
        for idx, person in enumerate(self.precompute_face_embedding_map):
            person_features = person.get("features")
            if metric == "euclidean":
                distance = euclidean(person_features, features)
            elif metric == "cosine":
                distance = cosine(person_features, features)
            name = person.get("name")
            if distance < min_dist_val:
                min_dist_val = distance
                min_dist_idx = idx
        if min_dist_val <= threshold:
            return self.precompute_face_embedding_map[min_dist_idx].get("name")
        else:
            return "Not recognized from database"
    
    def watch(self, url='https://www.youtube.com/watch?v=_IQOr7otBQU'):
        vPafy = pafy.new(url)
        play = vPafy.getbest(preftype="mp4")
        video_capture = cv2.VideoCapture(play.url)
        fps = FPS().start()
        
        prev_frame_time = time.time()
        while True:
            if not video_capture.isOpened():
                sleep(5)
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(64, 64)
            )
            face_imgs = np.empty((len(faces), 224, 224, 3))
            for i, face in enumerate(faces):
                face_img, cropped = self.processor.crop_face(frame, face, margin=10, size=224)
                (x, y, w, h) = cropped
                cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 200, 0), 2)
                face_imgs[i, :, :, :] = face_img
            if len(face_imgs) > 0:
                feature_faces = self.model.predict(face_imgs)
                predicted_names = [self.identify_face(feature_face) for feature_face in feature_faces]
            for i, face in enumerate(faces):
                label = f"{predicted_names[i]}"
                self.draw_label(frame, (face[0], face[1]), label)
            
            new_frame_time = time.time()
            cur_fps = 1/(new_frame_time-prev_frame_time)
            cur_fps = str(cur_fps)
            prev_frame_time = new_frame_time
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, cur_fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break 
            fps.update()
        
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        
        video_capture.release()
        cv2.destroyAllWindows()