import streamlit as st
from PyPDF2 import PdfReader
import docx2txt
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import RagTokenizer, RagSequenceForGeneration

# Set the page configuration with a black theme
st.set_page_config(
    page_title="Document-Chatbot",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the tokenizer and RAG model directly
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

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
def retrieve_relevant_passages(query, faiss_index, documents, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    relevant_passages = [documents[idx] for idx in indices[0]]
    return relevant_passages

# Function to generate response using RAG model
def generate_response(query, relevant_passages):
    try:
        if relevant_passages is None or len(relevant_passages) == 0:
            raise ValueError("No relevant passages found. Check your retrieval process.")
        
        # Combine relevant passages into a single string for tokenization
        context = " ".join(relevant_passages)
        st.write(f"Context used for generation: {context}")  # Debugging

        # Tokenize the query and context together
        inputs = tokenizer(query, text_pair=context, return_tensors='pt', padding=True, truncation=True)
        st.write(f"Tokenized inputs: {inputs}")  # Debugging
        
        if inputs is None or 'input_ids' not in inputs or 'attention_mask' not in inputs:
            raise ValueError("Tokenization failed. Ensure that the query is valid.")
        
        # Generate the response
        outputs = rag_model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        st.write(f"Generated output: {outputs}")  # Debugging
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        st.write(f"Error generating response: {e}")
        return "An error occurred while generating the response."

# Streamlit UI
st.title("Document-Chatbot")

st.write("Engage & Explore your documents.")

uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

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
        documents.append(text)
    
    faiss_index = create_faiss_index(documents)

    query = st.text_input("Ask a question:")
    if query:
        relevant_passages = retrieve_relevant_passages(query, faiss_index, documents)
        response = generate_response(query, relevant_passages)
        st.write(response)
