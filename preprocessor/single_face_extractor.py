import cv2
import os
import glob

from .constants import PROCESSED_FACE_FOLDER, RAW_IMAGE_FOLDER
from .frame_processor import FrameProcessor

class SingleFaceExtractor(object):
    """
    Singleton class to extraction face images from image files (each image only containing a person)
    """

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SingleFaceExtractor, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.processor = FrameProcessor()

    def extract_faces(self, raw_image_file, save_folder, save=True):
        img = cv2.imread(raw_image_file, cv2.IMREAD_COLOR)
        face_image, _ = processor.extract_faces(img)

        if face_image is not None:
            imgfile = os.path.basename(raw_image_file) + ".png"
            imgfile = os.path.join(save_folder, imgfile)
            cv2.imwrite(imgfile, face_image)

def main():
    extractor = SingleFaceExtractor()
    folders = list(glob.iglob(os.path.join(RAW_IMAGE_FOLDER, '*')))
    os.makedirs(PROCESSED_FACE_FOLDER, exist_ok=True)
    names = [os.path.basename(folder) for folder in folders]
    for i, folder in enumerate(folders):
        name = names[i]
        images = list(glob.iglob(os.path.join(folder, '*.*')))
        save_folder = os.path.join(PROCESSED_FACE_FOLDER, name)
        print(save_folder)
        os.makedirs(save_folder, exist_ok=True)
        for raw_image in images:
            extractor.extract_faces(raw_image, save_folder)
        print(f"done with {save_folder}")
    print("done preprocessing raw images into proper image!!!!")

if __name__ == "__main__":
    main()