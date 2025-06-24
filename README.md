# Chat with Your Notes — IBM Project

A Generative AI-powered chatbot that lets you upload a **PDF**, extract its text, and ask questions about the content. The answers are generated using **IBM Watsonx**, and the interface is built using **Streamlit**.

---

Deployed Link - https://prasadnotebot.streamlit.app/

## Features

- 📄 Upload and read content from any PDF
- 🤖 Ask questions based on the uploaded notes
- 🧠 Context-aware answers using IBM Watsonx
- 🌐 Beautiful, responsive UI with custom styling
- 🗂️ Vector-based indexing for accurate context retrieval

---

## Tech Stack

- **Python**
- **Streamlit** (Frontend UI)
- **IBM Watsonx** (Text Generation)
- **HuggingFace Transformers / Sentence Transformers** (For embeddings)
- **FAISS / Similar vector store** (For semantic search)
- **PyMuPDF / pdfplumber** (For PDF text extraction)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ShubhamPrasad3012/IBMProject.git
cd IBMProject
