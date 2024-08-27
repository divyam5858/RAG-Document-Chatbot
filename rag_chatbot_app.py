import streamlit as st
from PyPDF2 import PdfReader
import docx2txt
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import RagTokenizer, RagTokenForGeneration
import torch

# Set the page configuration
st.set_page_config(
    page_title="Document-Chatbot",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize RAG model and tokenizer
rag_model_name = "facebook/rag-token-nq"
rag_tokenizer = RagTokenizer.from_pretrained(rag_model_name)
rag_model = RagTokenForGeneration.from_pretrained(rag_model_name)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    return docx2txt.process(file)

# Function to extract text from TXT
def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Function to create FAISS index
def create_faiss_index(documents):
    document_embeddings = embedding_model.encode(documents)
    dimension = document_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(document_embeddings))
    return faiss_index

# Function to retrieve relevant passages
def retrieve_relevant_passages(query, faiss_index, documents, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    relevant_passages = [documents[idx] for idx in indices[0]]
    return relevant_passages

# Function to generate response using RAG model
def generate_response_via_rag(query, relevant_passages):
    try:
        if relevant_passages is None or len(relevant_passages) == 0:
            raise ValueError("No relevant passages found. Check your retrieval process.")
        
        # Ensure relevant_passages has exactly n_docs (5 by default)
        n_docs = 5
        if len(relevant_passages) < n_docs:
            relevant_passages.extend(relevant_passages[:n_docs - len(relevant_passages)])
        elif len(relevant_passages) > n_docs:
            relevant_passages = relevant_passages[:n_docs]
        
        # Combine relevant passages into a single string
        context = " ".join(relevant_passages)
        
        # Tokenize the inputs
        inputs = rag_tokenizer(query, return_tensors="pt")
        context_inputs = rag_tokenizer(context, return_tensors="pt", padding=True, truncation=True)

        # Generate response
        output = rag_model.generate(
            input_ids=inputs['input_ids'],
            context_input_ids=context_inputs['input_ids'].repeat_interleave(n_docs, dim=0),
            num_beams=5,
            early_stopping=True
        )
        response = rag_tokenizer.decode(output[0], skip_special_tokens=True)

        return response

    except Exception as e:
        st.write(f"Error generating response: {e}")
        return "An error occurred while generating the response."

# Streamlit UI
st.title("📄 Document-Chatbot")
st.write("Engage & Explore your documents.")

# Create columns with a 1:1 ratio
col1, col2 = st.columns([1, 1])

# Left column for uploading documents
with col1:
    st.subheader("Upload Your Documents")
    uploaded_files = st.file_uploader("Choose your documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    # Process uploaded files and create FAISS index
    documents = []
    if uploaded_files:
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(file)
            elif file.type == "text/plain":
                text = extract_text_from_txt(file)
            else:
                text = ""
            
            # Debugging: Show the extracted text for each document
            st.write(f"Extracted text from {file.name}:")
            st.write(text[:500])  # Show the first 500 characters for debugging
            
            documents.append(text)
        
        if documents:
            faiss_index = create_faiss_index(documents)
            st.session_state['faiss_index'] = faiss_index
            st.session_state['documents'] = documents
            st.success("Documents uploaded and processed successfully!")
        else:
            st.error("No valid text was extracted from the documents.")

# Right column for the main content (questions and responses)
with col2:
    st.subheader("Ask Your Question")
    query = st.text_input("Enter your question here:")
    if query and 'faiss_index' in st.session_state:
        faiss_index = st.session_state['faiss_index']
        documents = st.session_state['documents']
        
        # Debugging: Check if the FAISS index and documents are correct
        st.write("FAISS Index and documents are loaded.")
        
        relevant_passages = retrieve_relevant_passages(query, faiss_index, documents)
        
        # Debugging: Show the relevant passages found
        st.write("Relevant Passages Found:")
        for i, passage in enumerate(relevant_passages):
            st.write(f"Passage {i+1}: {passage[:500]}")  # Show the first 500 characters
        
        response = generate_response_via_rag(query, relevant_passages)
        st.write(response)
    elif not query:
        st.write("Please enter a question.")
    else:
        st.write("Please upload documents first.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Developed by Divyashree Mallarapu"
    "</div>",
    unsafe_allow_html=True,
)
