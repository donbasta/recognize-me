import cv2

class CCTVSimulator(object):
    """
    Class for simulating webcam as simple CCTV (IP Camera), and send frames to the server
    """

    def start(self):
        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            if not video_capture.isOpened():
                sleep(5)
            ret, frame = video_capture.read()
            cv2.imshow('CCTV', frame)
            #TODO: send frame here
            if cv2.waitKey(5) == 27: 
                break
        
        video_capture.release()
        cv2.destroyAllWindows()