import cv2

from ultralytics import YOLO
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
model=YOLO("model.pt")
while True:
    ret, frame=cam.read()
    if not ret:
        break
    results=model.predict(frame)
    box=results[0].plot() 
    cv2.putText(box, "hi", (100, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(16, 149, 194), 2)
    cv2.imshow("hello", box)
# Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
