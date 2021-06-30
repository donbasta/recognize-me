import os
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from mtcnn.mtcnn import MTCNN

class MultiFaceExtractor(object):
    """
    Given a single image with multiple person, extract and crop all of the faces of the given image
    """
    def __init__(self, filename, outdir=None, required_size=(224, 224), save=False):
        self.filename = filename
        self.outdir = outdir
        self.required_size = required_size
        self.save = save
    
    def extract_face(self):
        pixels = plt.imread(self.filename)

        detector = MTCNN()
        results = detector.detect_faces(pixels)

        faces_array = []

        for idx, res in enumerate(results):
            x1, y1, width, height = res['box']
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]

            image = Image.fromarray(face)
            image = image.resize(self.required_size)
            face_array = np.asarray(image)
            faces_array.append(face_array)

            if self.save:
                im = Image.fromarray(face_array)
                im.save(f'{self.outdir}/{idx}.jpg')
        
        return faces_array

if __name__ == "__main__":
    filename = input("insert filename: ")
    filename = f'data/raw_images/{filename}'
    folder_path = input("insert folder to save the images in faces folder: ")
    outdir = f'data/processed_face/{folder_path}'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    extractor = MultiFaceExtractor(filename=filename, outdir=outdir)
    extractor.extract_face()