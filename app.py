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

st.set_page_config(page_title="NeuralDoc")
st.title("NeuralDoc")

# session state initialization
if "messages" not in st.session_state:
  st.session_state.messages = []

if "vector_store" not in st.session_state:
  st.session_state.vector_store = None

with st.sidebar:
# uploading the file
  uploaded_file = st.file_uploader("Upload your pdf",type="pdf")

  # reading the content
  if uploaded_file and st.session_state.vector_store is None:
    # Save the file temporarily so we can read it
    temp_file = "./temp.pdf"
    with open(temp_file,"wb") as file:
      file.write(uploaded_file.getvalue())

    try:
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
      st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)

      st.success("Vector Store Created! I am ready to search.")

    except Exception as e:
      st.error(f"Error processing PDF: {e}")

  if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.vector_store = None
    st.rerun()

# Re-draw all previous messages
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# ask a query
query = st.chat_input("Ask your question about this pdf")

if query:
  # if pdf not uploaded
  if st.session_state.vector_store is None:
    st.error("Please upload a PDF file first!")

  else:
    #Save user question to memory
    st.session_state.messages.append({"role":"user", "content": query})

    #show user message
    with st.chat_message("user"):
      st.markdown(query)

    #generate answer
    with st.chat_message("assistant"):
      message_placeholder = st.empty()
      message_placeholder.markdown("Thinking...")

      try:
        #configuring retriever
        retriever = st.session_state.vector_store.as_retriever(
          search_type = "similarity_score_threshold",
          search_kwargs = {"k":3, "score_threshold":0.3}
        )

        retrieved_docs = retriever.invoke(query)

        with st.expander("View Source Context"):
          for i,doc in enumerate(retrieved_docs):
            st.write(f"Chunk {i+1}: {doc.page_content[:200]}")

        if not retrieved_docs:
          response_text = "Couldn't find any relevant text in the PDF for the given query."
        else:
          # create context string
          context = " ".join([doc.page_content for doc in retrieved_docs])

          model = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

          # send to AI using manual prompting
          messages = [
            {"role": "system", "content":"You are a helpful assistant. Answer ONLY using the provided cotext accurately. If the answer is not present, say you do not know."},
            {"role":"user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
          ]

          response = model.invoke(messages)
          response_text = response.content

        message_placeholder.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

      except Exception as e:
        st.write(f"Error during retrieval. The question might be too different from the PDF content. Details: {e}")

  