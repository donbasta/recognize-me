import argparse

def main():
    parser = argparse.ArgumentParser("AMS")
    parser.add_argument("-m", "--mode", type=str, default="local-stream")
    parser.add_argument("-e", "--embedding", type=str, default="./data/embeddings/tes.pickle")
    parser.add_argument("-y", "--youtube_url", type=str, default="https://www.youtube.com/watch?v=_IQOr7otBQU", help="Youtube Video Link")
    parser.add_argument("-h", "--host", type=str, default="127.0.0.1", help="Host for API")
    parser.add_argument("-p", "--port", type=int, default=5000, help="Port for API")
    args = vars(parser.parse_args())

    mode = args["mode"]
    embedding = args["embedding"]

    if mode == "local-stream":
        from app.FaceRecognitionStream import FaceRecognitionStream
        face = FaceRecognitionStream(embedding=embedding)
        face.detect_face()
    
    if mode == "youtube-stream":
        from app.YoutubeStream import YoutubeStream
        url = args["youtube_url"]
        youtube = YoutubeStream(embedding=embedding)
        youtube.watch(url=url)
    
    if mode == "server":
        from api.index import app
        host = args["host"]
        port = args["port"]
        print(f"server berjalan pada host {host} and port {port}")
        app.run(host=host, port=port)
    
    if mode == "train-all":
        pass

if __name__ == "__main__":
    main()