from ultralytics import YOLO
import cv2
import os
model=YOLO("model.pt")
def analyze_video(video_path):
    stream=cv2.VideoCapture(video_path) #setup stream to analyze each frame of video
    #setup output video
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path="static/output/test.mp4"
    fps=30
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width,height))
    #analyze each frame
    while True:
        ret, frame=stream.read()
        if ret:
            result=model(frame)
            out.write(result)
            
        else:
            break
    stream.release()
    out.release()