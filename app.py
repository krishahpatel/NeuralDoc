# To fix Python 3.13 crash
import concurrent.futures.process
import concurrent.futures.thread

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# uploading the file
uploaded_file = st.file_uploader("Upload your pdf",type="pdf")

# reading the content
if uploaded_file is not None:
  # Save the file temporarily so we can read it
  temp_file = "./temp.pdf"
  with open(temp_file,"wb") as file:
    file.write(uploaded_file.getvalue())

  st.write("Reading PDF...")

  loader = PyPDFLoader(temp_file)
  docs = loader.load()

  # making chunks
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
  )
  chunks = text_splitter.split_documents(docs)

  st.write(f"Splitting done. Total chunks: {len(chunks)}")
  
  # generating embeddings
  st.write("Generating Embeddings......(This might take a minute)")

  embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

  # creating vector store
  vector_store = FAISS.from_documents(chunks, embeddings)

  st.success("Vector Store Created! I am ready to search.")

  # ask a query
  query = st.text_input("Ask your question about this pdf")

  if query:
    #configuring retriever
    retriever = vector_store.as_retriever(
      search_type = "similarity_score_threshold",
      search_kwargs = {"k":3, "score_threshold":0.3}
    )

    # fetch the relevant docs manually
    try:
      retrieved_docs = retriever.invoke(query)

      with st.expander("See what the AI found(Debug info)"):
        for i,doc in enumerate(retrieved_docs):
          st.write(f"Chunk {i+1}: {doc.page_content[:200]}")

      if not retrieved_docs:
        st.warning("Couldn't find any relevant text in the PDF for the given query.")
      else:
        # create context string
        context = " ".join([doc.page_content for doc in retrieved_docs])

        model = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

        # send to AI using manual prompting
        messages = [
          {"role": "system", "content":"You are a helpful assistant. Use the provided context to answer the user's question accurately."},
          {"role":"user", "content": f"Context:\n\nQuestion:\n{query}"}
        ]

        st.write("Thinking...")
        response = model.invoke(messages)

        st.write(response.content)

    except Exception as e:
      st.write(f"Error during retrieval. The question might be too different from the PDF content. Details: {e}")