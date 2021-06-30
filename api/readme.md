# API Documentation

## adding several images of a person
### request body: (from user interface / frontend) 
- raw images (must be a picture containing a single person only)
- name/NIM/identity of the person (unique, if it is exist already in the database will be ignored)
- save=boolean (whether to save the picture or not), 
- retrain=boolean (retrain the model using the new images or not)


## processing frames of videos

### request body: (from camera)
- video stream (or frame by frame)

### response: (to backend server)
- processed frames (with bounding box)
- for each frame, list of the bounding box coordinates and the prediction result