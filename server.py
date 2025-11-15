from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from analyze import analyze_video
from data_manager import DataManager
import uuid
app=Flask(__name__)#creates a flask server on variable app
app.config["UPLOAD_FOLDER"]="uploads"
OUTPUT_IMAGE="static/output.jpg"
model=YOLO("model.pt")
SERVICE_ACCOUNT_PATH = "service_account.json"
FIREBASE_API_KEY = "AIzaSyARs60WidQXyaLfwjnfV5xfb6iYIMWohQI"  # from Firebase Console > Project Settings > Web API Key
VIDEO_DB = {}
data_manager = DataManager(SERVICE_ACCOUNT_PATH, FIREBASE_API_KEY)
@app.route("/upload", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            video_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], video_id)
            file.save(filepath)
            # Run YOLO
            analyze_video(filepath)
            # Store mapping
            VIDEO_DB[video_id] = filename


            return jsonify({
                "message": "Upload successful",
                "video_id": video_id
            }), 200
    return render_template("index.html")


    


@app.route('/video/<video_id>', methods=['GET'])
def get_video(video_id):
    if video_id not in VIDEO_DB:
        return jsonify({"error": "Video not found"}), 404

    filename = VIDEO_DB[video_id]
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)
@app.route("/auth/register", methods=["POST"])
def register():
    body = request.json
    email = body.get("email")
    password = body.get("password")
    try:
        user = data_manager.register_user(email, password)
        return jsonify(user), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
@app.route("/auth/login", methods=["POST"])
def login():
    body = request.json
    email = body.get("email")
    password = body.get("password")
    try:
        tokens = data_manager.login_user(email, password)
        return jsonify(tokens), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
@app.route("/auth/logout", methods=["POST"])
def logout():
    try:
        id_token = request.json.get("idToken")
        if not id_token:
            return jsonify({"error": "Missing idToken"}), 400

        result = data_manager.sign_out_user(id_token)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
@app.route("/user/data", methods=["GET", "POST", "PATCH", "DELETE"])
def user_data():
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return jsonify({"error": "Missing Authorization header"}), 401
    try:
        token = auth_header.split(" ")[1]  # "Bearer <idToken>"
        decoded = data_manager.verify_user(token)
        uid = decoded["uid"]
        if request.method == "POST":
            body = request.json
            body["userId"]=uid
            data_manager.create_document("videos", "", body)
            return jsonify({"status": "created"}), 201
        elif request.method == "GET":
            # fetch all documents with userId == uid
            docs = data_manager.query_collection("videos", "userId", "==", uid)
            return jsonify(docs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 401



if __name__=="__main__": 
    app.run(debug=True, host="0.0.0.0", port=5000)