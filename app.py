import streamlit as st
from pdf_utils import extract_text_from_pdf
from vector_store import build_vector_index, get_top_chunks
from watsonx_client import generate_answer

st.title("🧠 Chat with Your Notes")

uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf:
    st.success("PDF uploaded. Extracting text...")
    text = extract_text_from_pdf(uploaded_pdf)
    index, chunks, model = build_vector_index(text)

    user_question = st.text_input("Ask a question:")

    if user_question:
        context = get_top_chunks(user_question, model, index, chunks)
        answer = generate_answer(user_question, context)
        st.write("### 💬 Answer:")
        st.write(answer)
