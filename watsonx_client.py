import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
REGION = "us-south"  

print("🔐 API key loaded:", API_KEY is not None)
print("📁 Project ID loaded:", PROJECT_ID is not None)

def get_ibm_token():
    url = "https://iam.cloud.ibm.com/identity/token"
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": API_KEY,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    res = requests.post(url, data=data, headers=headers)

    try:
        res.raise_for_status()
        return res.json()["access_token"]
    except Exception as e:
        print("❌ Failed to get IBM access token")
        # Shubham Prasad
        print("Status code:", res.status_code)
        print("Response:", res.text)
        raise e

def generate_answer(question, context):
    access_token = get_ibm_token()

    prompt = f"""Provide a detailed, structured, and comprehensive answer based on the following notes:

Notes:
{context}

Question: {question}
Answer:"""

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    body = {
        "model_id": "google/flan-ul2",
        "input": prompt,
        "project_id": PROJECT_ID,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 400
        }
    }

    url = f"https://{REGION}.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    res = requests.post(url, headers=headers, json=body)

    print("📡 Request sent to WatsonX.")
    print("🧾 Request body:", body)
    print("🛬 Status Code:", res.status_code)
    print("🔍 Response Text:", res.text)

    try:
        res.raise_for_status()
        response_json = res.json()
        return response_json["results"][0]["generated_text"]
    except Exception as e:
        print("❌ Error during generation")
        raise e
