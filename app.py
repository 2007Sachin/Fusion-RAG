import streamlit as st
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from config import GROQ_API_KEY, MODEL_NAME, EMBEDDING_MODEL_NAME

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Fusion RAG Chatbot")
st.title("🔀 Fusion RAG Chatbot")

# Load LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load_and_split()
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # FAISS Retriever
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2
    
    # Fusion Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.5, 0.5]
    )
    
    # Build QA Chain with Fusion Retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        chain_type="stuff", 
        retriever=ensemble_retriever
    )

    # Chat Interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = qa_chain.invoke(user_input)
            st.markdown(response["result"])
        st.session_state.chat_history.append({"role": "assistant", "content": response["result"]})