import os
import faiss
import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from streamlit_option_menu import option_menu

# Groq API Key
client = Groq(api_key="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")

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

# FAISS Index
dimension = 384  
index = faiss.IndexFlatL2(dimension)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "Research Article Helper", 
        ["Home", "Upload PDF", "Summary", "About"], 
        icons=["house", "file-earmark-arrow-up", "file-earmark-text", "info-circle"], 
        menu_icon="cast", 
        default_index=0,
    )

# App title
st.title("📚 Research Article Helper")
st.write("Interact with research articles by uploading PDFs and asking questions.")

# "Upload PDF" Section
if selected == "Upload PDF":
    st.subheader("📂 Upload a PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)

        # Encode chunks and store embeddings in FAISS
        embeddings = embedding_model.encode(chunks)
        for i, embedding in enumerate(embeddings):
            index.add(np.array([embedding]))

        st.success(f"✅ Processed {len(chunks)} chunks and stored embeddings in FAISS.")
        st.session_state['chunks'] = chunks

# "Summary" Section
if selected == "Summary":
    st.subheader("📑 Get a Summary of the Uploaded PDF")
    if 'chunks' in st.session_state:
        if st.button("Generate Summary"):
            full_text = " ".join(st.session_state['chunks'][:5])  # Summarizing first 5 chunks as an example
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Provide a concise summary of the following text:\n\n{full_text}",
                    }
                ],
                model="llama3-8b-8192",
            )
            summary = chat_completion.choices[0].message.content
            st.write("### 📋 Summary:")
            st.write(summary)
        else:
            st.info("Click the button above to generate a summary.")
    else:
        st.warning("Please upload a PDF in the 'Upload PDF' section first.")

# "About" Section
if selected == "About":
    st.subheader("ℹ️ About")
    st.write("**Research Article Helper** is a tool to interact with research articles by uploading PDFs. Use it to ask questions, generate summaries, and gain insights from scientific documents.")
    st.write("Built using Streamlit, FAISS, and Groq AI.")