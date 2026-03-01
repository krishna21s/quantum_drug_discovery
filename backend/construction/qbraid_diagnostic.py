import requests
import os

API_KEY = os.getenv("5wsijtkjk135guw5weow57o4a9j01y")  # safer than hardcoding
JOB_QRN = "qbraid:job:abc123xyz"

url = f"https://api-v2.qbraid.com/api/v1/jobs/{JOB_QRN}"

headers = {
    "X-API-KEY": API_KEY
}

response = requests.get(url, headers=headers)

print(response.json())