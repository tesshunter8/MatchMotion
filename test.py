from google.oauth2 import service_account
from google.cloud import firestore
import os, time

# If using service account JSON:
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="service_account.json"

from flask import Flask, jsonify
from google.cloud import firestore
import requests

app = Flask(__name__)

@app.route("/ip")
def ip():
    try:
        r = requests.get("https://ipinfo.io/json", timeout=10)
        return jsonify({"requests_ip": r.json()})
    except Exception as e:
        return jsonify({"requests_error": str(e)}), 500

@app.route("/rest_check")
def rest_check():
    """Check REST access to Google services (these URLs are valid)."""
    urls = [
        "https://www.google.com",
        "https://firebase.google.com",
        "https://www.googleapis.com/discovery/v1/apis",
        "https://firestore.googleapis.com/$discovery/rest?version=v1",
        "https://storage.googleapis.com/storage/v1/projects"  # Will return 403 if reachable
    ]

    results = {}

    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            # 403 means we **reached Google**, just not authorized — that's SUCCESS for the test
            results[url] = r.status_code
        except Exception as e:
            results[url] = f"ERROR: {type(e).__name__}: {str(e)}"

    return jsonify(results)



@app.route("/grpc_check")
def grpc_check():
    try:
        from google.oauth2 import service_account
        from google.cloud import firestore_v1

        creds = service_account.Credentials.from_service_account_file(r"C:\Users\lenovo\Desktop\python\webcam\service_account.json")
        db = firestore.Client(project="matchmotion-56add", credentials=creds)

        # 5s timeout to avoid indefinite hang
        docs = list(db.collection("test_connection_check").limit(1).stream(timeout=5))

        return jsonify({"grpc_ok": True, "docs_count": len(docs)})

    except Exception as e:
        return jsonify({"grpc_error": str(e)}), 500
if __name__ == "__main__":
    app.run(port=5000, debug=True)