import streamlit as st
from pdf_utils import extract_text_from_pdf
from vector_store import build_vector_index, get_top_chunks
from watsonx_client import generate_answer

st.markdown("""
<style>
body {
    margin: 0;
    padding: 0;
    background: linear-gradient(to right top, #ff9a9e, #fad0c4, #fad0c4, #fbc2eb, #a6c1ee);
    background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
    color: #222;
}

/* Fix Streamlit base padding */
.stApp {
    padding: 2rem 1rem;
    background: transparent;
}

/* Gradient Header */
.header {
    background: linear-gradient(to right, #f857a6, #ff5858);
    padding: 1.8rem 2rem;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 2.4rem;
    font-weight: 800;
    box-shadow: 0 12px 25px rgba(248, 87, 166, 0.4);
    margin-bottom: 2rem;
    transition: all 0.3s ease;
}

/* Glassmorphism Containers */
.section {
    background: rgba(255, 255, 255, 0.45);
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.4);
    margin-bottom: 2rem;
}

.answer-box {
    background: linear-gradient(to right, #ffecd2, #fcb69f);
    color: blue;
    padding: 1.5rem;
    font-weight: 500;
    border-radius: 14px;
    font-size: 1.1rem;
    box-shadow: 0 6px 15px rgba(255, 138, 101, 0.2);
    animation: fadeIn 0.5s ease-in-out;
}

/* Animations */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Input tweaks */
input, textarea {
    background-color: white !important;
    border-radius: 10px !important;
    border: 1px solid #ccc !important;
    color: #222 !important;
}
</style>
""", unsafe_allow_html=True)
 
st.markdown("<div class='header'>📝 Chat with Your Notes shubham </div>", unsafe_allow_html=True)
 
st.markdown("<div class='section'>", unsafe_allow_html=True)
uploaded_pdf = st.file_uploader("📄 Upload your PDF Notes", type="pdf")
st.markdown("</div>", unsafe_allow_html=True)
 
if uploaded_pdf:
    st.success("✅ PDF uploaded. Extracting text...")
    text = extract_text_from_pdf(uploaded_pdf)
    index, chunks, model = build_vector_index(text)

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("### 🤔 Ask something about your notes:")
    with st.form(key="qa_form"):
        user_question = st.text_input("Your question...", placeholder="e.g., Summarize the uploaded notes")
        submit_btn = st.form_submit_button("Submit")
    st.markdown("</div>", unsafe_allow_html=True)

    if submit_btn and user_question:
        with st.spinner("🔍 Thinking..."):
            context = get_top_chunks(user_question, model, index, chunks, k=5)
            answer = generate_answer(user_question, context)

        st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
        st.markdown("### 💬 Answer:")
        st.markdown(f"<p>{answer}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
st.markdown(""" 
<hr style="margin-top: 3rem; border: 0.5px solid #eee;">
<p style="text-align:center; font-size: 0.85rem; color: #666;">
🚀 Made with ❤️ using IBM Watsonx & Streamlit<br>
<span style="font-weight:600;">– by Shubham Prasad</span>
</p>
""", unsafe_allow_html=True)
