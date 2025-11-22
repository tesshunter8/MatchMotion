import requests
import firebase_admin
from firebase_admin import credentials, auth
import urllib


class DataManager:
    def __init__(self, service_account_path, api_key, project_id):
        cred = credentials.Certificate(service_account_path)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        self.api_key = api_key
        self.project_id = project_id

        self.firestore_base = (
            f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents"
        )

        self.query_url = (
            f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents:runQuery"
        )

        self.storage_url = (
            f"https://firebasestorage.googleapis.com/v0/b/{project_id}.appspot.com/o?uploadType=media&name="
        )

    # -----------------------------
    # Authentication (same as yours)
    # -----------------------------
    def register_user(self, email: str, password: str):
        try:
            user = auth.create_user(email=email, password=password)
            return {"uid": user.uid, "email": user.email}
        except Exception as e:
            raise ValueError(f"Error creating user: {e}")

    def login_user(self, email: str, password: str):
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self.api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}

        r = requests.post(url, json=payload)
        if r.status_code == 200:
            return r.json()
        else:
            raise ValueError(f"Login failed: {r.json()}")

    def verify_user(self, id_token: str):
        try:
            decoded_token = auth.verify_id_token(id_token)
            return decoded_token
        except Exception as e:
            raise ValueError(f"Invalid authentication token: {e}")

    def logout_user(self, id_token):
        try:
            decoded_token = self.verify_user(id_token)
            uid = decoded_token["uid"]
            auth.revoke_refresh_tokens(uid)
            return {"status": "success", "message": "User signed out."}
        except Exception as e:
            raise ValueError(f"Error signing out user: {e}")

    # -----------------------------
    # Helpers
    # -----------------------------
    def encode_fields(self, data: dict):
        """Convert Python dict → Firestore REST fields"""
        fields = {}
        for key, value in data.items():
            if isinstance(value, bool):
                fields[key] = {"booleanValue": value}
            elif isinstance(value, int):
                fields[key] = {"integerValue": str(value)}
            elif isinstance(value, float):
                fields[key] = {"doubleValue": value}
            elif isinstance(value, dict):
                fields[key] = {"mapValue": {"fields": self.encode_fields(value)}}
            else:
                fields[key] = {"stringValue": str(value)}
        return fields

    # -----------------------------
    # Firestore REST: Create / Set
    # -----------------------------
    def create_document(self, collection, doc_id, data):
        fields = {"fields": self.encode_fields(data)}

        # ---------------------
        # AUTO-ID MODE
        # ---------------------
        if not doc_id:
            url = f"{self.firestore_base}/{collection}"
            r = requests.post(url, json=fields)   # POST to auto-create
            if r.status_code not in (200, 201):
                raise ValueError(f"Firestore POST error: {r.text}")
            response = r.json()
            created_id = response["name"].split("/")[-1]
            return {"status": "success", "doc_id": created_id}

        # ---------------------
        # SET DOCUMENT BY ID
        # ---------------------
        url = f"{self.firestore_base}/{collection}/{doc_id}"
        r = requests.patch(url, json=fields)

        if r.status_code not in (200, 201):
            raise ValueError(f"Firestore PATCH error: {r.text}")

        return {"status": "success", "doc_id": doc_id}
    # -----------------------------
    # Firestore REST: Query
    # -----------------------------
    def query_collection(self, collection: str, field: str, op: str, value):
        """
        Example:
        query_collection("users", "userId", "==", uid)
        """
        # Map your operator to Firestore REST operator
        op_map = {
            "==": "EQUAL",
            ">": "GREATER_THAN",
            "<": "LESS_THAN",
            ">=": "GREATER_THAN_OR_EQUAL",
            "<=": "LESS_THAN_OR_EQUAL",
        }

        if op not in op_map:
            raise ValueError("Unsupported operator for Firestore REST API")

        structured_query = {
            "structuredQuery": {
                "from": [{"collectionId": collection}],
                "where": {
                    "fieldFilter": {
                        "field": {"fieldPath": field},
                        "op": op_map[op],
                        "value": self.encode_fields({"v": value})["v"],
                    }
                }
            }
        }

        r = requests.post(self.query_url, json=structured_query)

        if r.status_code != 200:
            raise ValueError(f"Query error: {r.text}")

        results = []
        for item in r.json():
            if "document" in item:
                doc = item["document"]
                doc_id = doc["name"].split("/")[-1]
                fields = doc.get("fields", {})
                results.append({"id": doc_id, "fields": fields})

        return results
    def upload_to_storage(self, file_path):
        file_binary=open(file_path, "rb").read()
        headers = { "Content-Type": "video/mp4" }
        encoded_url=urllib.parse.quote(file_path, safe='')
        url=self.storage_url+"uploads%2FClip.mp4"
        
        print (url)
        res=requests.post(url, data=file_binary, headers=headers)
        response=res.json()
        return response
