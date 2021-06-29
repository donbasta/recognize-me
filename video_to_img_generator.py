import cv2
import os
import glob
import pickle
from keras_vggface.vggface import VGGFace
import numpy as np
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image

from constants import PROCESSED_FACE_FOLDER, VIDEOS_FOLDER, HAARCASCADE_FRONTALFACE_ALT, HAARCASCADE_FRONTALFACE_DEFAULT

class FaceExtractor(object):
    """
    Singleton class to extraction face images from video files
    """
    CASE_PATH = [HAARCASCADE_FRONTALFACE_ALT, HAARCASCADE_FRONTALFACE_DEFAULT]

    def __new__(cls, weight_file=None, face_size=224):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceExtractor, cls).__new__(cls)
        return cls.instance

    def __init__(self, face_size=224):
        self.face_size = face_size

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

    def extract_faces(self, video_file, save_folder):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH[0])

        # 0 means the default video capture device in OS
        cap = cv2.VideoCapture(video_file)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        print("length: {}, w x h: {} x {}, fps: {}".format(length, width, height, fps))
        # infinite loop, break by key ESC
        frame_counter = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame_counter = frame_counter + 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=10,
                    minSize=(64, 64)
                )
                # only keep the biggest face as the main subject
                face = None
                if len(faces) > 1:  # Get the largest face as main face
                    face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))  # area = w * h
                elif len(faces) == 1:
                    face = faces[0]
                if face is not None:
                    face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                    (x, y, w, h) = cropped
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    cv2.imshow('Faces', frame)
                    imgfile = os.path.basename(video_file).replace(".","_") +"_"+ str(frame_counter) + ".png"
                    imgfile = os.path.join(save_folder, imgfile)
                    cv2.imwrite(imgfile, face_img)
            if cv2.waitKey(5) == 27:  # ESC key press
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()

def main():
    model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    extractor = FaceExtractor()
    folders = list(glob.iglob(os.path.join(VIDEOS_FOLDER, '*')))
    os.makedirs(PROCESSED_FACE_FOLDER, exist_ok=True)
    names = [os.path.basename(folder) for folder in folders]
    for i, folder in enumerate(folders):
        name = names[i]
        videos = list(glob.iglob(os.path.join(folder, '*.*')))
        save_folder = os.path.join(PROCESSED_FACE_FOLDER, name)
        print(save_folder)
        os.makedirs(save_folder, exist_ok=True)
        for video in videos:
            extractor.extract_faces(video, save_folder)

if __name__ == "__main__":
    main()

