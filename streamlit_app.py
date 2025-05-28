import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import os

UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Flan-T5 model
@st.cache_resource
def load_flan_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Load and split document
def load_and_split(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

# Build FAISS index
def create_faiss_index(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# Search document chunks
def search_docs(index, query):
    results = index.similarity_search(query, k=3)
    return "\n\n".join([r.page_content for r in results])

# Streamlit App UI
st.title("📄 Chat with Your Document (LangChain + Flan-T5)")

# Initialize session state
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "flan" not in st.session_state:
    st.session_state.flan = None

# Upload file
if not st.session_state.file_uploaded:
    uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'docx', 'txt'])
    if uploaded_file:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.file_uploaded = True
        st.session_state.file_path = file_path

        st.success("✅ Document uploaded and being processed...")

        docs = load_and_split(file_path)
        st.session_state.faiss_index = create_faiss_index(docs)
        st.session_state.flan = load_flan_pipeline()
        st.success("🎯 Ready! Ask questions about your document:")

# Clear button
if st.session_state.file_uploaded:
    if st.button("🧹 Clear Document"):
        try:
            os.remove(st.session_state.file_path)
        except Exception as e:
            st.warning(f"Failed to delete file: {e}")
        st.session_state.file_uploaded = False
        st.session_state.file_path = None
        st.session_state.faiss_index = None
        st.session_state.flan = None
        st.experimental_rerun()

# Question input
if st.session_state.file_uploaded:
    user_input = st.text_input("Ask a question:")
    if user_input:
        context = search_docs(st.session_state.faiss_index, user_input)
        prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
        response = st.session_state.flan(prompt, max_new_tokens=256)[0]['generated_text']

        st.markdown("### 🧠 Answer")
        st.write(response)
