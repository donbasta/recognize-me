import os
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from mtcnn.mtcnn import MTCNN

def extract_face(filename, outdir=None, required_size=(224, 224), save=False):
    pixels = plt.imread(filename)

    detector = MTCNN()
    results = detector.detect_faces(pixels)

    faces_array = []

    for idx, res in enumerate(results):
        x1, y1, width, height = res['box']
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]

        image = Image.fromarray(face)
        image = image.resize((224, 224))
        face_array = np.asarray(image)
        faces_array.append(face_array)

        if save:
            im = Image.fromarray(face_array)
            im.save(f'{outdir}/{idx}.jpg')
    
    return faces_array

if __name__ == "__main__":
    filename = input("insert filename: ")
    filename = f'data/test_images/{filename}'
    folder_path = input("insert folder to save the images in faces folder: ")
    outdir = f'data/extracted_faces/{folder_path}'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    extract_face(filename=filename, outdir=outdir)