from ultralytics import YOLO
import cv2
import os
from pathlib import Path
model=YOLO("models/people10.pt")
modelshuttle=YOLO("models/bestshuttle.pt")
modelcourt=YOLO("models/courtseg50.pt")
STOP_THRESHOLD_FRAMES=15 #number of frames needed to determine point/fault of shuttle
BOUNDING_BOX_OFFSET=5

def draw_path(frame, path, color):
    for coord in path:
        cv2.circle(frame, coord, 10, color, 1)
    return frame
def check_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)
def track_shuttle(detections, shuttle_box, stop_counter):
    stop=False
    for (x1,y1,x2,y2,conf,name) in detections:
        if name=="shuttle":
            overlap=check_overlap(shuttle_box, (x1, y1, x2, y2))
            #print (f"overlap:{overlap}")
            if overlap >= 0.1: 
                stop_counter+=1
                print (stop_counter)
                if stop_counter>=STOP_THRESHOLD_FRAMES:
                    print ("shuttle stopped moving")
                    stop=True
            else: 
                shuttle_box=(x1-BOUNDING_BOX_OFFSET,y1-BOUNDING_BOX_OFFSET,x2+BOUNDING_BOX_OFFSET,y2+BOUNDING_BOX_OFFSET)
                stop_counter=0
    return shuttle_box, stop_counter, stop

def shuttle_point(court_bounds, shuttle_pos):
    x=int((shuttle_pos[0]+shuttle_pos[2])//2)
    y=int((shuttle_pos[1]+shuttle_pos[3])//2)
    masks = court_bounds.masks.data  # list of N masks (each shape [H, W])
    names = court_bounds.names
    boxes = court_bounds.boxes
    regions = []
    
    for i, mask in enumerate(masks):
        cls_id = int(boxes.cls[i].item())
        label = names[cls_id]
        conf = boxes.conf[i].item()
        print (label)
        # Convert mask to numpy
        mask_np = mask.cpu().numpy()

        # Check if shuttle center (x, y) is inside this mask
        #print(f"y: {y}, x: {x}, mask_np.shape[0]: {mask_np.shape[0]}, mask_np.shape[1]: {mask_np.shape[1]}, mask_np[y,x]: {mask_np[y,x]}")
        if y < mask_np.shape[0] and x < mask_np.shape[1]:
        
            if mask_np[y, x] > 0.5:  # pixel belongs to region
                regions.append((label, conf))

    if regions:
        # If shuttle center is inside one or more regions
        print(f"Shuttle is inside region(s): {[r[0] for r in regions]}")
    else:
        print("Shuttle is not inside any known region.")
    isAboveNet(court_bounds, shuttle_pos)

def isAboveNet(court_bounds, object_pos):
    x=int((object_pos[0]+object_pos[2])//2)
    y=int((object_pos[1]+object_pos[3])//2)

    for d in court_bounds.boxes:
        x1, y1, x2, y2 = d.xyxy[0].tolist()
        conf = d.conf[0].item()
        cls_id = int(d.cls[0].item())
        label = court_bounds.names[cls_id]  # class name
        if label == "net":
            if y>y2:
                return False
            else:
                return True
def shot_number(top_hits, bottom_hits, player_pos, shuttle_pos, court_bounds):
    if check_overlap(player_pos,shuttle_pos)>0.5:
        if isAboveNet(court_bounds, player_pos): 
            top_hits+=1
        else: 
            bottom_hits+=1
    return top_hits, bottom_hits 
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
    #get framecount for progress
    framestotal=int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    frames=0
    #set up shuttle tracking 
    shuttle_box=(0,0,0,0)
    stop_counter=0
    #court bounds
    court_bounds={}
    #player positions
    top_pos=[]
    bottom_pos=[]
    shuttle_pos=[]
    #track player shuttle hits
    top_hits=0
    bottom_hits=0
    #analyze each frame
    while True:
        ret, frame=stream.read()
        if ret:
            frames+=1
            print (f"progress:{((frames/framestotal)*100):.1f}%")
            # Run inference
            results_player = model(frame, conf=0.3, verbose=False)
            results_shuttle = modelshuttle(frame, conf=0.1, verbose=False)
            if frames==1:
                results_court=modelcourt(frame, conf=0.3, verbose=False)[0]
                court_bounds=results_court
                court_frame=results_court.plot()
                cv2.imwrite("court.png", court_frame)
            detections=[]
            # Draw detections directly on the frame
            for result in [results_player[0], results_shuttle[0]]:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                names = result.names

                for (x1, y1, x2, y2, conf, cls) in zip(*[boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], confs, classes]):
                    name=names[cls]
                    if name=="Shuttle - v1 2025-09-13 11-11am":
                        name="shuttle"
                    detections.append((x1,y1,x2,y2,conf,name))
            merged_detections=[]
            shuttle_in_frame=()
            person_in_frame=()
            for d in sorted(detections, key=lambda x:x[4], reverse=True):
                x1,y1,x2,y2,conf,name=d
                if name == "shuttle":
                    # check if overlaps with an already kept shuttle box
                    overlap = False
                    for md in merged_detections:
                        if md[5] == "shuttle" and check_overlap((x1, y1, x2, y2), (md[0], md[1], md[2], md[3])) > 0.5:
                            overlap = True
                            shuttle_in_frame=(md[0], md[1], md[2], md[3])
                            break
                    if overlap:
                        continue  # skip duplicate
                if name=="person":
                    person_in_frame=(x1,y1,x2,y2)
                    x=int((x1+x2)//2)
                    y=int((y1+y2)//2)
                    if isAboveNet(court_bounds, (x1, y1, x2, y2)):
                        top_pos.append((x, y))
                    else:
                        bottom_pos.append((x, y))
                
                merged_detections.append(d)
            if shuttle_in_frame and person_in_frame:
                top_hits, bottom_hits=shot_number(top_hits, bottom_hits, person_in_frame, shuttle_in_frame, court_bounds)
                print ("hits:",top_hits, bottom_hits)
            shuttle_box, stop_counter, stop=track_shuttle(merged_detections, shuttle_box, stop_counter)
            s=shuttle_box
            x=int((s[0]+s[2])//2)
            y=int((s[1]+s[3])//2)
            shuttle_pos.append((x,y))
            merged_detections.append((s[0],s[1],s[2],s[3],1,"shuttle_box"))
            for (x1,y1,x2,y2,conf,name) in merged_detections: 
                color = (0, 255, 0) if name == "shuttle" else (255, 0, 0)
                label = f"{name} {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            #combines different layers of detection
            if stop:
                cv2.putText(
                    frame,                   # image
                    "Shuttle Stopped Moving!",           # text
                    (50, 100),                # position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX, # font
                    1,                        # font scale
                    (0, 255, 0),              # color (B, G, R)
                    2,                        # thickness
                    cv2.LINE_AA               # anti-alias for smoother text
                )
                shuttle_point(court_bounds, shuttle_box)
            draw_path(frame, top_pos, (0, 0, 255))
            draw_path(frame, bottom_pos, (255, 0, 0))
            draw_path(frame, shuttle_pos, (0,0,0))
            out.write(frame)
        else:
            break
    stream.release()
    out.release()