from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image

import numpy as np
import os
import pickle
import glob

from constants import PROCESSED_FACE_FOLDER

def picklify(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()

class Embedder(object):
    def __init__(self, folder_path, outdir="../data/embeddings/tes.pickle"):
        self.model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        self.folder = folder_path
        self.folders = list(glob.iglob(os.path.join(self.folder, '*')))
        self.names = [os.path.basename(folder) for folder in self.folders]
        self.outdir = outdir
    
    def image2x(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        pixels = image.img_to_array(img)
        pixels = np.expand_dims(pixels, axis=0)
        pixels = preprocess_input(pixels, version=2)
        return pixels

    def cal_mean_feature(self, image_folder):
        face_images = list(glob.iglob(os.path.join(image_folder, '*.*')))

        def chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i:min(i + n, len(l) - 1)]

        batch_size = 32
        face_images_chunks = chunks(face_images, batch_size)
        fvecs = None
        for face_images_chunk in face_images_chunks:
            images = np.concatenate([self.image2x(face_image) for face_image in face_images_chunk])
            batch_fvecs = self.model.predict(images)
            if fvecs is None:
                fvecs = batch_fvecs
            else:
                fvecs = np.append(fvecs, batch_fvecs, axis=0)
        return np.array(fvecs).sum(axis=0) / len(fvecs)

    def get_embeddings(self, save=True, outdir='default'):
        embeddings = []
        for i, folder in enumerate(self.folders):
            name = self.names[i]
            print(f"processing {name}.....")
            img_folder = os.path.join(PROCESSED_FACE_FOLDER, name)
            mean_features = self.cal_mean_feature(image_folder=img_folder)
            embeddings.append({
                "name": name,
                "features": mean_features
            })
        picklify(self.outdir, embeddings)
        print("done!")

if __name__ == "__main__":
    embedder = Embedder(PROCESSED_FACE_FOLDER)
    embedder.get_embeddings()