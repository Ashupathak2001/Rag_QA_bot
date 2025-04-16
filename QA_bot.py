# QA_bot.py (TensorFlow Version)
import streamlit as st
import tempfile
import os
import sys
from server import RAGModel

def init_rag_model():
    """Initialize the RAG model with API keys."""
    # Try multiple methods to get the API key
    cohere_api_key = None
    
    # Method 1: From Streamlit secrets
    try:
        cohere_api_key = st.secrets["COHERE_API_KEY"]
        st.success("Found API key in Streamlit secrets!")
    except KeyError:
        st.warning("API key not found in Streamlit secrets.")
    
    # Method 2: From environment variables
    if not cohere_api_key:
        cohere_api_key = os.environ.get("COHERE_API_KEY")
        if cohere_api_key:
            st.success("Found API key in environment variables!")
    
    # Method 3: From user input
    if not cohere_api_key:
        st.info("Please enter your Cohere API key below:")
        cohere_api_key = st.text_input("Cohere API Key:", type="password")
        if not cohere_api_key:
            st.error("No API key provided. Cannot initialize the model.")
            return None
    
    # Initialize the model with the API key
    try:
        return RAGModel(cohere_api_key=cohere_api_key)
    except Exception as e:
        st.error(f"Error initializing RAG model: {str(e)}")
        st.info(f"Exception details: {str(e)}")
        return None

def main():
    st.title("RAG Based Document Q&A Bot")
    
    # Initialize session state
    if 'rag_model' not in st.session_state:
        with st.spinner("Initializing RAG model..."):
            st.session_state.rag_model = init_rag_model()
        
        if st.session_state.rag_model is None:
            st.warning("RAG model initialization failed. Please check the API key and try again.")
            st.stop()  # Stop execution if model initialization failed
    
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    
    # Add a clear index button
    if st.button("Clear Index"):
        st.session_state.rag_model.clear_index()
        st.session_state.document_processed = False
        st.success("Index cleared successfully!")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])
    
    if uploaded_file:
        with st.spinner('Processing document...'):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process and index the document
            try:
                num_chunks = st.session_state.rag_model.index_document(tmp_file_path)
                st.session_state.document_processed = True
                st.success(f"Document processed successfully! ({num_chunks} chunks indexed)")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                st.info(f"Exception details: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
    
    # Question input
    question = st.text_input("Ask a question about your document:")
    
    if question:
        if not st.session_state.document_processed:
            st.warning("Please upload and process a document first.")
        else:
            with st.spinner('Generating answer...'):
                try:
                    response = st.session_state.rag_model.query(question)
                    
                    # Display answer
                    st.write("### Answer:")
                    st.write(response["answer"])
                    
                    # Display relevant contexts
                    with st.expander("View relevant contexts"):
                        for i, (context, distance) in enumerate(zip(response["contexts"], response["distances"]), 1):
                            st.write(f"Context {i} (Distance: {distance:.4f}):")
                            st.write(context)
                            st.write("---")
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    st.info(f"Exception details: {str(e)}")

if __name__ == "__main__":
    # Set this environment variable to reduce warnings from TensorFlow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    # Launch the app
    main()