from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from analyze import analyze_video
from data_manager import DataManager
import uuid
from datetime import datetime

app=Flask(__name__)#creates a flask server on variable app
app.config["UPLOAD_FOLDER"]="uploads"
OUTPUT_IMAGE="static/output.jpg"
model=YOLO("model.pt")
SERVICE_ACCOUNT_PATH = "service_account.json"
FIREBASE_API_KEY = "AIzaSyARs60WidQXyaLfwjnfV5xfb6iYIMWohQI"  # from Firebase Console > Project Settings > Web API Key
data_manager = DataManager(SERVICE_ACCOUNT_PATH, FIREBASE_API_KEY, "matchmotion-56add")
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
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], video_id)
            file.save(filepath)
            # Run YOLO
            results=analyze_video(filepath)

            # attach uid and name from app
            results["userId"] = (data_manager.verify_user(request.form.get("userId")))["uid"]
            results["name"] = request.form.get("name")
            results["createdAt"] = datetime.now().isoformat()

            response=data_manager.upload_to_storage(results["output_path"])
            print (response)
            results["url"]=response["url"]
            print (results)
            # Store mapping
            data_manager.create_document("videos", "", results)

            return jsonify({
                "message": "Upload successful",
                "video_id": video_id
            }), 200
    return render_template("index.html")

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
@app.route("/user/data", methods=["GET", "POST"])
def user_data():
    # -----------------------------
    # Auth
    # -----------------------------
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return jsonify({"error": "Missing Authorization header"}), 401

    try:
        token = auth_header.split(" ")[1]  # Expect "Bearer <idToken>"
        decoded = data_manager.verify_user(token)
        uid = decoded["uid"]

        # -----------------------------
        # CREATE DOCUMENT
        # -----------------------------
        if request.method == "POST":
            body = request.json
            if not body:
                return jsonify({"error": "Missing request body"}), 400

            body["userId"] = uid
            result = data_manager.create_document("videos", "", body)  # auto-generate doc ID
            return jsonify({"status": "created", "doc_id": result["doc_id"]}), 201
        elif request.method == "GET":
            docs = data_manager.query_collection("videos", "userId", "==", uid)
            # Optional: decode Firestore fields into plain Python dicts
            decoded_docs = []
            for doc in docs:
                fields = {}
                for k, v in doc["fields"].items():
                    # decode basic types
                    if "stringValue" in v:
                        fields[k] = v["stringValue"]
                    elif "integerValue" in v:
                        fields[k] = int(v["integerValue"])
                    elif "doubleValue" in v:
                        fields[k] = float(v["doubleValue"])
                    elif "booleanValue" in v:
                        fields[k] = v["booleanValue"]
                    elif "mapValue" in v:
                        # optional: flatten mapValue recursively
                        fields[k] = v["mapValue"]["fields"]
                decoded_docs.append({"id": doc["id"], **fields})
        return jsonify(decoded_docs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 401

if __name__=="__main__": 
    app.run(debug=True, host="0.0.0.0", port=5000)