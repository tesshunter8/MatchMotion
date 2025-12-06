import requests

# Your bucket
bucket = "matchmotion-56add.firebasestorage.app"

# REST endpoint for upload
url = f"https://firebasestorage.googleapis.com/v0/b/{bucket}/o"

# Local file to upload
file_path = "test.txt"

# Storage path you want
destination = "testing/test-upload.txt"

# MUST be URL-encoded (slashes → %2F)
from urllib.parse import quote
destination_encoded = quote(destination, safe="")

params = {
    "uploadType": "media",
    "name": destination_encoded
}

# Read file
with open(file_path, "rb") as f:
    file_data = f.read()

# Send request
r = requests.post(url, params=params, data=file_data)

print("Status:", r.status_code)
print("Response:", r.text)