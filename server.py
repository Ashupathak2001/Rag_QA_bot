# server.py (TensorFlow Version)
import os
import cohere
import numpy as np
import faiss
import json
from dotenv import load_dotenv
from typing import List, Dict, Any
import PyPDF2
import tempfile

class DocumentProcessor:
    def __init__(self):
        # Import TensorFlow modules locally to avoid conflicts
        import tensorflow as tf
        import tensorflow_hub as hub
        tf.config.set_visible_devices([], 'GPU')
        
        # Load Universal Sentence Encoder from TF Hub
        self.encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    def process_pdf(self, pdf_file: str) -> List[str]:
        """Extract text from PDF and split into chunks."""
        chunks = []
        with open(pdf_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                # Split into smaller chunks (you can adjust the chunk size)
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                for paragraph in paragraphs:
                    # Further split long paragraphs if needed
                    if len(paragraph) > 512:
                        words = paragraph.split()
                        for i in range(0, len(words), 100):
                            chunk = ' '.join(words[i:i + 100])
                            if chunk:
                                chunks.append(chunk)
                    else:
                        chunks.append(paragraph)
        return chunks

    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks using TensorFlow."""
        # Universal Sentence Encoder creates 512-dimensional embeddings
        embeddings = self.encoder(chunks)
        
        # Convert to numpy array
        return embeddings.numpy()

class FAISSIndex:
    def __init__(self, dimension: int = 512):  # Universal Sentence Encoder uses 512 dimensions
        """Initialize FAISS index."""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        
    def add_vectors(self, vectors: np.ndarray, chunks: List[str]):
        """Add vectors and their corresponding text chunks to the index."""
        self.index.add(vectors)
        self.chunks.extend(chunks)
    
    def search(self, query_vector: np.ndarray, k: int = 3) -> tuple:
        """Search for similar vectors and return their distances and indices."""
        return self.index.search(query_vector.reshape(1, -1), k)
    
    def save(self, index_file: str, chunks_file: str):
        """Save the FAISS index and chunks to disk."""
        faiss.write_index(self.index, index_file)
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f)
    
    def load(self, index_file: str, chunks_file: str):
        """Load the FAISS index and chunks from disk."""
        self.index = faiss.read_index(index_file)
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

class RAGModel:
    def __init__(self, cohere_api_key: str):
        load_dotenv()
        
        # Initialize FAISS index
        self.index = FAISSIndex()
        
        # Initialize Cohere client
        self.co = cohere.Client(cohere_api_key)
        
        # Initialize document processor
        self.doc_processor = DocumentProcessor()
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        self.index_file = 'data/faiss.index'
        self.chunks_file = 'data/chunks.json'
        
        # Load existing index if available
        if os.path.exists(self.index_file) and os.path.exists(self.chunks_file):
            self.index.load(self.index_file, self.chunks_file)
    
    def index_document(self, pdf_path: str) -> int:
        """Process and index a document."""
        # Process PDF into chunks
        chunks = self.doc_processor.process_pdf(pdf_path)
        
        if not chunks:
            raise ValueError("No text could be extracted from the PDF")
        
        # Create embeddings
        embeddings = self.doc_processor.create_embeddings(chunks)
        
        # Add to FAISS index
        self.index.add_vectors(embeddings, chunks)
        
        # Save the updated index
        self.index.save(self.index_file, self.chunks_file)
        
        return len(chunks)
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Query the RAG model with a question."""
        # Create query embedding
        import tensorflow as tf
        import tensorflow_hub as hub
        
        # Reuse the same encoder from document processor
        query_embedding = self.doc_processor.encoder([question]).numpy()[0]
        
        # Query FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get relevant contexts
        contexts = [self.index.chunks[idx] for idx in indices[0]]
        
        # Generate answer using Cohere
        prompt = f"""Based on the following contexts, answer the question.
        
Contexts:
{' '.join(contexts)}

Question: {question}

Answer:"""
        
        response = self.co.generate(
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
        
        return {
            "answer": response.generations[0].text.strip(),
            "contexts": contexts,
            "distances": distances[0].tolist()
        }

    def clear_index(self):
        """Clear the current index and saved files."""
        self.index = FAISSIndex()
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.chunks_file):
            os.remove(self.chunks_file)