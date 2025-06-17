import requests
import os
from dotenv import load_dotenv
load_dotenv()


API_KEY = os.getenv("WATSONX_API_KEY")  # Store in env variable
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")  # Get from your project in watsonx
REGION = "us-south"  # or "eu-de", etc.

def get_ibm_token():
    url = "https://iam.cloud.ibm.com/identity/token"
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": API_KEY,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    res = requests.post(url, data=data, headers=headers)
    return res.json()["access_token"]

def generate_answer(question, context):
    access_token = get_ibm_token()
    prompt = f"""Answer the question based on the notes:

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
            "max_new_tokens": 150
        }
    }

    url = f"https://{REGION}.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    res = requests.post(url, headers=headers, json=body)
    return res.json()["results"][0]["generated_text"]
