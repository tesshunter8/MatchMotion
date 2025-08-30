from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from analyze import analyze_video
app=Flask(__name__)#creates a flask server on variabel app
app.config["UPLOAD_FOLDER"]="uploads"
OUTPUT_IMAGE="static/output.jpg"
model=YOLO("model.pt")

@app.route("/", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            # Run YOLO
            analyze_video(filepath)
    return render_template("index.html")


if __name__=="__main__": 
    app.run(debug=True)