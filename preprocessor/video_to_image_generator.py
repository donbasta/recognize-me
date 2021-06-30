import cv2
import os
import glob

from .constants import PROCESSED_FACE_FOLDER, VIDEOS_FOLDER
from .frame_processor import FrameProcessor

class VideoToImagesExtractor(object):
    """
    Singleton class to extraction face images from video files
    """

    def __new__(cls, weight_file=None, face_size=224):
        if not hasattr(cls, 'instance'):
            cls.instance = super(VideoToImagesExtractor, cls).__new__(cls)
        return cls.instance

    def __init__(self, face_size=224):
        self.face_size = face_size
        self.processor = FrameProcessor()

    def extract_faces(self, video_file, save_folder):
        cap = cv2.VideoCapture(video_file)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        print("length: {}, w x h: {} x {}, fps: {}".format(length, width, height, fps))
        frame_counter = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame_counter = frame_counter + 1
                face_image, cropped = self.processor.extract_faces(frame)
                if face_image is not None:
                    (x, y, w, h) = cropped
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    cv2.imshow('Faces', frame)
                    imgfile = os.path.basename(video_file).replace(".","_") +"_"+ str(frame_counter) + ".png"
                    imgfile = os.path.join(save_folder, imgfile)
                    cv2.imwrite(imgfile, face_image)
            if cv2.waitKey(5) == 27:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break
        cap.release()
        cv2.destroyAllWindows()

def main():
    extractor = VideoToImagesExtractor()
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

