from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import time
import math
import pickle
import logging
import argparse

from raw_img_processor import FaceExtractor
from utils import decode_image

ALLOWED_IMAGE_EXTENSIONS = {'jpg','jpeg','png'}

app = Flask("Face Recognition API")
log = logging.getLogger(__name__)

extractor = FaceExtractor()

@app.route("/", methods=["GET"])
def test():
    data = {"message": "API can be accessed!", "status": "server running"}
    return jsonify(data), 200

@app.route("/add", methods=["POST"])
def add_image_face():
    """
    Adding person face to server database
    :param
    frame: a CV2 read image of the person
    id: unique id of the person (name, nim) 
    save: whether to save the picture or not
    """

    try:
        img = decode_image(request.files["image"].read())
    except Exception as e:
        log.error(e)
        data = {"error": "Error while loading image"}
        return jsonify(data), 500
    save_picture = False
    if request.args.get("save") == "true":
        save_picture = True
    
    face_img = extractor.extract_faces()

@app.route("/predict", methods=["POST"])
def process_frame():
    """
    Processing the video frames (predict + give bounding box + send to backend)
    """
    return "OK"

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Face Recognition API")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for API")
    parser.add_argument("--port", type=int, default=5000, help="Host for API")
    args = vars(parser.parse_args())
    host = args["host"]
    port = args["port"]
    print(f"erver running on host {host} and port {port}")
    app.run(host=host, port=port)