import firebase_admin
from firebase_admin import credentials, firestore, auth
import requests



class DataManager: 
    def __init__(self, service_account_path, api_key):
        cred=credentials.Certificate(service_account_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        self.db=firestore.client()
        self.api_key=api_key
    def register_user(self, email: str, password: str):
        """
        Register a new Firebase Authentication user.
        """
        try:
            user = auth.create_user(email=email, password=password)
            return {"uid": user.uid, "email": user.email}
        except Exception as e:
            raise ValueError(f"Error creating user: {e}")
    def login_user(self, email: str, password: str):
        """
        Login a user using Firebase Authentication REST API.
        Returns idToken and refreshToken.
        """
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self.api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        r = requests.post(url, json=payload)
        if r.status_code == 200:
            return r.json()
        else:
            raise ValueError(f"Login failed: {r.json()}")
    def verify_user(self, id_token: str):
        """
        Verifies a Firebase Auth ID token.
        Returns decoded token if valid.
        """
        try:
            decoded_token = auth.verify_id_token(id_token)
            return decoded_token
        except Exception as e:
            raise ValueError(f"Invalid authentication token: {e}")
    def create_document(self, collection, doc_id, data):
        if doc_id:
            # use custom document id
            self.db.collection(collection).document(doc_id).set(data)
        else:
            # let Firestore generate a random document id
            self.db.collection(collection).add(data)
        return {"status": "success", "doc_id": doc_id or "auto-generated"}
    def query_collection(self, collection: str, field: str, op: str, value: str):
        """
        Query documents in a Firestore collection with a where filter.
        Example: query_collection("users", "userId", "==", uid)
        """
        try:
            docs = self.db.collection(collection).where(field, op, value).stream()
            results = []
            for doc in docs:
                results.append({"id": doc.id, **doc.to_dict()})
            return results
        except Exception as e:
            raise ValueError(f"Error querying collection: {e}")