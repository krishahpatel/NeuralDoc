import streamlit as st
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 

# uploading the file
uploaded_file = st.file_uploader("Upload your pdf",type="pdf")

# reading the content
if uploaded_file is not None:
  pdf_reader = PyPDF2.PdfReader(uploaded_file)

  text = ""
  for page in pdf_reader.pages:
    text += page.extract_text()

  # making chunks
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
  )
  chunks = text_splitter.split_text(text)

  st.write(f"Splitting done. Total chunks: {len(chunks)}")
  
  # generating embeddings
  st.write("Generating Embeddings......(This might take a minute)")

  embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

  # creating vector store
  vector_store = FAISS.from_texts(chunks, embedding = embeddings)

  st.success("Vector Store Created! I am ready to search.")

  # ask a query
  query = st.text_input("Ask your question about this pdf")

  if query:
    docs = vector_store.similarity_search(query = query, k = 3)

    st.write("Found these relevant chunks:")

    for doc in docs:
      st.write(doc.page_content)
      st.write("-------")