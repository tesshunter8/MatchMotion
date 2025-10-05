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
           
            # Run inference
            results_player = model(frame, conf=0.3, verbose=False)
            results_shuttle = modelshuttle(frame, conf=0.1, verbose=False)

            # Draw detections directly on the frame
            for result in [results_player[0], results_shuttle[0]]:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                names = result.names

                for (x1, y1, x2, y2, conf, cls) in zip(*[boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], confs, classes]):
                    label = f"{names[cls]} {conf:.2f}"
                    color = (0, 255, 0) if names[cls] == "shuttle" else (255, 0, 0)  # different color per model
                    # Draw bounding box and label
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            #combines different layers of detection
            out.write(frame)
        else:
            break
    stream.release()
    out.release()