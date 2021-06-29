* video_data: To generate raw_faces from video footage, put video (consisting of only 1 person face) to video folder

* raw_img: inserted manually (upload 1 pics)

* test_face: testing feature embedding p3.npy

* processed_face: converted result from raw_img or video frames to (224, 224) dimension for embedding calculation

* extractor_test: test multi person images (generate one image for each person, and calculate each of their embedding)

* extractor_extracted_face: faces extraction from images in extractor_test using FaceExtractor

* embeddings: face embeddings of each person --> would be best if grouped by sections (named using sectionID)
	--> API for adding a person embeddings in a section
	--> API for deleting a person embeddings in a section
	--> API for updating a person embeddings in a section


# FLOW

video -> processed_face
raw_img -> processed_face
processed_face -> embeddings