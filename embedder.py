from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions

import matplotlib.pyplot as plt
import numpy as np
import os

vgg_senet50_model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def get_embeddings(folder_faces, save=True, outdir='default'):
    
    faces = [plt.imread(os.path.join(folder_faces, facepath)) for facepath in os.listdir(folder_faces)]

    # faces = []
    # for facepath in os.listdir(folder_faces):
    #     print("facepath: ", facepath)
    #     path = os.path.join(folder_faces, facepath)
    #     faces.append(plt.imread(path))
    #     plt.imshow(plt.imread(path))
    #     plt.show()

    samples = np.asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    features = vgg_senet50_model.predict(samples)
    if save:
        np.save(f'{outdir}.npy', features)
    return features

if __name__ == "__main__":
    folder_faces = input("enter path of the face images: ")
    outdir = input("enter path for the face embeddings: ")
    outdir = f'data/embeddings/{outdir}'
    embeddings = get_embeddings(folder_faces, outdir=outdir)