from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine

import matplotlib.pyplot as plt
import numpy as np

from face_extractor import extract_face

vgg_senet50_model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def is_match(known_embedding, candidate_embedding, threshold=0.5):
    score = cosine(known_embedding, candidate_embedding)
    if score <= threshold:
        print('face is a match (%.3f <= %.3f)' % (score, threshold))
    else:
        print('face is not a match (%.3f > %.3f)' % (score, threshold))

def recognizer(facepath, facedata):
    # take the first picture only, assume only one face per image
    pixels = extract_face(facepath)
    if len(pixels) == 0:
        print("Not detecting a face in this image, quitting...")
        return
    pixels = pixels[0]
    plt.imshow(pixels)
    plt.show()

    pixels = pixels.astype('float32')

    samples = np.expand_dims(pixels, axis=0)
    samples = preprocess_input(samples, version=2)
    prediction = vgg_senet50_model.predict(samples)
    for idx, f in enumerate(facedata):
        print(f"{idx}-th image: ")
        is_match(prediction, f, threshold=0.3)

if __name__ == "__main__":
    # facepath = input("enter path of the face image you wish to recognize: ")
    facepaths = [   
                    './data/test_face/farras1.jpg', 
                    './data/test_face/farras2.jpg', 
                    './data/test_face/farras3.jpg', 
                    './data/test_face/farras4.jpg',
                    './data/test_face/afif1.jpg',
                    './data/test_face/bos1.jpg',
                ]
    # facedata = input("enter path of the embedding files you want to search ")
    facedata = './data/embeddings/p3.npy'
    facedata = np.load(facedata)
    for idx, facepath in enumerate(facepaths):
        print(f"testcase {idx}:")
        recognizer(facepath, facedata)