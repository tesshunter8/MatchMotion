import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    ret, frame=cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) #puts image through model, attempts to detect face
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 255, 136), 10)
    cv2.putText(frame, "hi", (100, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(16, 149, 194), 2)
    cv2.imshow("hello", frame)
# Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
