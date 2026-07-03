import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def generate_answer(question, context):

    prompt = f"""
You are an AI assistant for answering questions from uploaded PDF documents.

Instructions:
- Answer ONLY using the provided context.
- If the answer is not present in the context, say:
  "I couldn't find that information in the uploaded document."
- Give detailed and well-structured answers whenever possible.
- Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text