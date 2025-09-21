from ultralytics import YOLO
import cv2
import os
from pathlib import Path
model=YOLO("best50.pt")
modelshuttle=YOLO("bestshuttle.pt")
def analyze_video(video_path):
    video_stem=Path(video_path).stem
    stream=cv2.VideoCapture(video_path) #setup stream to analyze each frame of video
    #setup output video
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path="static/output"
    os.makedirs(output_path, exist_ok=True)
    output_video=Path(output_path)/f"{video_stem}.mp4"
    fps=30
    out = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width,height))
    #analyze each frame
    while True:
        ret, frame=stream.read()
        if ret:
            results1=model(frame, conf=0.3)
            results2=modelshuttle(frame, conf=0.1)
            results1frame=results1[0].plot()
            results2frame=results2[0].plot()
            results1frame=cv2.addWeighted(results1frame, 1, results2frame, 1, 0)
            out.write(results1frame)
            
        else:
            break
    stream.release()
    out.release()