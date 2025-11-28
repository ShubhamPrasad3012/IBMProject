import os
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()

API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

def generate_answer(question, context):
    """Generate answer using watsonx.ai foundation model"""
   
    # Create credentials
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": API_KEY
    }
   
    # Set parameters
    parameters = {
        GenParams.MAX_NEW_TOKENS: 300,
        GenParams.DECODING_METHOD: "greedy",
        GenParams.TEMPERATURE: 0.7,
        GenParams.TOP_P: 1,
        GenParams.TOP_K: 50
    }
   
    # Initialize the model
    model = Model(
        model_id="ibm/granite-3-3-8b-instruct",
        params=parameters,
        credentials=credentials,
        project_id=PROJECT_ID
    )
   
    # Create the prompt
    prompt = f"""Provide a detailed, structured, and comprehensive answer based on the given notes.

Notes:
{context}

Question: {question}

Answer:"""
   
    # Generate response
    response = model.generate_text(prompt=prompt)
   
    return response