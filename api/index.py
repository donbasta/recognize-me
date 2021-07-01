from flask import Flask, request, jsonify
import logging
import argparse
import os

from PIL import Image
from preprocessor.frame_processor import FrameProcessor
from .utils import decode_image

ALLOWED_IMAGE_EXTENSIONS = {'jpg','jpeg','png'}

app = Flask("Face Recognition API")
log = logging.getLogger(__name__)

processor = FrameProcessor()
face_uploads_dir = "../data/raw_images" if __name__ == "__main__" else "./data/raw_images"

@app.route("/", methods=["GET"])
def test():
    data = {"message": "API can be accessed!", "status": "server running"}
    return jsonify(data), 200

@app.route("/test-add-image", methods=["POST"])
def test_add_image():
    images = request.files.getlist("images")
    # name = request.args.get
    name = request.form.get('name')
    print(f"name: {name}")
    savedir = os.path.join(face_uploads_dir, name)
    print("path to save the file: ", savedir)
    print("ADA GA? ", os.path.exists(savedir))
    os.makedirs(savedir, exist_ok=True)
    print(f"number of images uploaded: {len(images)}")
    for image in images:
        print(f"name: {image.filename} -- {os.path.join(savedir, image.filename)}")
        image.save(os.path.join(savedir, image.filename))
    data = {"saveto": f"{savedir}", "status": "OK"}
    return jsonify(data), 200

@app.route("/add-image", methods=["POST"])
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
    
    face_img, _ = processor.extract_faces()
    #TODO
    #1. get several images if possible
    #2. save face_img array as picture if save_picture == True
    #3. pipe face_img array to embedder --> embedder needs to be modified to not from a folder, but from array of face_img
    #4. get the embedder result, insert to a pickle object --> can be section ID, or whatever

@app.route("/predict", methods=["POST"])
def process_frame():
    """
    Processing the video frames (predict + give bounding box + send to backend)
    """
    return "OK"