import streamlit as st
from pdf_utils import extract_text_from_pdf
from vector_store import build_vector_index, get_top_chunks
from watsonx_client import generate_answer
 
st.markdown("""
<style>
@keyframes fadeIn {
  from {opacity: 0;}
  to {opacity: 1;}
}

body {
  background: linear-gradient(to right, #141e30, #243b55);
  color: white;
}

h1, h2, h3 {
  color: #f4f4f4;
}
</style>
""", unsafe_allow_html=True)
 
st.markdown("""
<div style='
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    padding: 1rem 2rem;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 1.8rem;
    font-weight: bold;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    margin-bottom: 2rem;
'>
🧠 Chat with Your Notes
</div>
""", unsafe_allow_html=True)
 
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
 
uploaded_pdf = st.file_uploader("📄 Upload a PDF", type="pdf")

if uploaded_pdf:
    st.success("✅ PDF uploaded. Extracting text...")
    text = extract_text_from_pdf(uploaded_pdf)
    index, chunks, model = build_vector_index(text)

    st.markdown("### 🤔 Ask something about your notes:")
 
    with st.form(key="qa_form"):
        user_question = st.text_input("Your question...", placeholder="e.g., give summary of pdf")
        submit_btn = st.form_submit_button("Submit")
 
    if submit_btn and user_question:
        with st.spinner("🔍 Thinking..."):
            context = get_top_chunks(user_question, model, index, chunks, k=5)
            answer = generate_answer(user_question, context)

        st.markdown("### 💬 Answer:")
        st.markdown(
            f"""
            <div style='
                background: rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 1rem;
                color: #f1f1f1;
                font-size: 1.05rem;
                line-height: 1.6;
                animation: fadeIn 0.6s ease-in-out;
                border: 1px solid rgba(255,255,255,0.15);
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
            '>
                {answer}
            </div>
            """,
            unsafe_allow_html=True
        )
 
st.markdown("""
<hr style="margin-top: 3rem;">
<small>
Made with ❤️ using IBM Watsonx & Streamlit
</small>
""", unsafe_allow_html=True)
