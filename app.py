import streamlit as st
import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings.base import Embeddings


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0]


# Load models
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = SentenceTransformer('all-MiniLM-L12-v2')

# Initialize FAISS index
embedding_dim = bi_encoder.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Initialize SemanticChunker with SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
semantic_chunker = SemanticChunker(embeddings=embeddings)

# Function to load and chunk text files


def load_and_chunk_txt_files(txt_files):
    texts = []
    doc_names = []
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append(text)
            # Extract the file name without the extension
            doc_name = os.path.splitext(os.path.basename(txt_file))[0]
            doc_names.append(doc_name)

    # Apply Semantic Chunking
    all_chunks = []
    chunk_indices = []
    for i, text in enumerate(texts):
        chunks = semantic_chunker.split_text(text)
        all_chunks.extend(chunks)
        chunk_indices.extend([i] * len(chunks))

    return all_chunks, chunk_indices, doc_names

# Function to save embeddings


def save_embeddings(embeddings, filename="embeddings.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)
    # Comment out or remove the following line to suppress success message
    # st.success(f"Les embeddings ont été sauvegardés dans {filename}.")

# Function to load embeddings


def load_embeddings(filename="embeddings.pkl"):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Load and process text files
st.title("Système de Recherche Sémantique")
txt_files = ["https://github.com/SAFAALHAKKAOUI/semantic_engine/blob/main/pdf1.txt",
             "https://github.com/SAFAALHAKKAOUI/semantic_engine/blob/main/pdf2.txt"]

with st.spinner("Traitement des fichiers..."):
    chunks, chunk_indices, doc_names = load_and_chunk_txt_files(txt_files)
    chunk_embeddings = bi_encoder.encode(chunks, convert_to_tensor=True)

    # Add embeddings to FAISS index
    faiss_index.add(chunk_embeddings.cpu().numpy())

    # Option to save embeddings
    save_embeddings(chunk_embeddings.cpu().numpy(),
                    filename="chunk_embeddings.pkl")

    # Comment out or remove the following line to suppress success message
    # st.success("Fichiers chargés et traités avec succès!")

# Input query
query = st.text_input("Entrez votre requête ici :")
if query:
    with st.spinner("Recherche en cours..."):
        # Reformater le vecteur d'embeddings de la requête pour avoir une dimension (1, d)
        query_embedding = bi_encoder.encode(
            query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

        # Recherche sémantique avec FAISS (Top 5 résultats)
        distances, indices = faiss_index.search(query_embedding, k=5)
        retrieved_chunks = [chunks[idx] for idx in indices[0]]
        retrieved_docs = [doc_names[chunk_indices[idx]] for idx in indices[0]]

        # Rerank using cross-encoder
        cross_input = [query] * len(retrieved_chunks)
        chunk_embeddings = cross_encoder.encode(
            retrieved_chunks, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(
            query_embedding, chunk_embeddings).numpy().flatten()
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # Display reranked results
        st.subheader("Résultats:")
        for idx in sorted_indices:
            st.write(f"**Document:** {retrieved_docs[idx]}")
            st.write(f"**Chunk:** {retrieved_chunks[idx]}")
            st.write(f"**Score de similarité:** {similarity_scores[idx]:.4f}")
            st.write("---")
