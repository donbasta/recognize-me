import argparse

from app.FaceRecognitionStream import FaceRecognitionStream

def main():
    parser = argparse.ArgumentParser("Face Recognition Video Stream")
    parser.add_argument("--embedding", type=str, default="./data/embeddings/tes.pickle")
    args = vars(parser.parse_args())
    embedding = args["embedding"]
    face = FaceRecognitionStream(embedding=embedding)
    face.detect_face()

if __name__ == "__main__":
    main()