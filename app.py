import streamlit as st
import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings.base import Embeddings

# Classe pour utiliser SentenceTransformer dans le SemanticChunker
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0]

# Chargement des modèles
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = SentenceTransformer('all-MiniLM-L12-v2')

# Initialisation de l'index FAISS
embedding_dim = bi_encoder.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Initialisation du SemanticChunker avec SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
semantic_chunker = SemanticChunker(embeddings=embeddings)

# Vérification des fichiers dans le répertoire "dataa"
directory = "dataa"
try:
    files_in_directory = os.listdir(directory)
    st.write(f"Fichiers disponibles dans {directory} : ", files_in_directory)
except FileNotFoundError as e:
    st.error(f"Le répertoire {directory} est introuvable. Assurez-vous que le chemin est correct.")
    st.write(f"Erreur : {e}")
except Exception as e:
    st.error(f"Une erreur inattendue est survenue lors de l'accès au répertoire {directory}.")
    st.write(f"Erreur : {e}")

# Fonction pour charger et découper les fichiers texte
def load_and_chunk_txt_files(txt_files):
    texts = []
    doc_names = []
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append(text)
            # Extraire le nom de fichier sans l'extension
            doc_name = os.path.splitext(os.path.basename(txt_file))[0]
            doc_names.append(doc_name)

    # Appliquer le découpage sémantique
    all_chunks = []
    chunk_indices = []
    for i, text in enumerate(texts):
        chunks = semantic_chunker.split_text(text)
        all_chunks.extend(chunks)
        chunk_indices.extend([i] * len(chunks))

    return all_chunks, chunk_indices, doc_names

# Fonction pour sauvegarder les embeddings
def save_embeddings(embeddings, filename="embeddings.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)
    st.success(f"Les embeddings ont été sauvegardés dans {filename}.")

# Fonction pour charger les embeddings
def load_embeddings(filename="embeddings.pkl"):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Interface utilisateur
st.title("Système de Recherche Sémantique")

# Charger et traiter les fichiers texte
txt_files = ["dataa/pdf1.txt", "dataa/pdf2.txt"]

with st.spinner("Traitement des fichiers..."):
    try:
        chunks, chunk_indices, doc_names = load_and_chunk_txt_files(txt_files)
        chunk_embeddings = bi_encoder.encode(chunks, convert_to_tensor=True)

        # Ajouter les embeddings à l'index FAISS
        faiss_index.add(chunk_embeddings.cpu().numpy())

        # Option pour sauvegarder les embeddings
        save_embeddings(chunk_embeddings.cpu().numpy(), filename="chunk_embeddings.pkl")
        st.success("Fichiers chargés et traités avec succès!")
    except FileNotFoundError as e:
        st.error("Erreur lors du chargement des fichiers texte. Assurez-vous que les fichiers existent.")
        st.write(f"Erreur : {e}")
    except Exception as e:
        st.error("Une erreur inattendue est survenue lors du traitement des fichiers.")
        st.write(f"Erreur : {e}")

# Saisie de la requête
query = st.text_input("Entrez votre requête ici :")
if query:
    with st.spinner("Recherche en cours..."):
        try:
            # Embedding de la requête
            query_embedding = bi_encoder.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

            # Recherche sémantique avec FAISS (Top 5 résultats)
            distances, indices = faiss_index.search(query_embedding, k=5)
            retrieved_chunks = [chunks[idx] for idx in indices[0]]
            retrieved_docs = [doc_names[chunk_indices[idx]] for idx in indices[0]]

            # Rerank en utilisant le cross-encoder
            cross_input = [query] * len(retrieved_chunks)
            chunk_embeddings = cross_encoder.encode(retrieved_chunks, convert_to_tensor=True)
            similarity_scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings).numpy().flatten()
            sorted_indices = np.argsort(similarity_scores)[::-1]

            # Afficher les résultats rerankés
            st.subheader("Résultats:")
            for idx in sorted_indices:
                st.write(f"**Document:** {retrieved_docs[idx]}")
                st.write(f"**Chunk:** {retrieved_chunks[idx]}")
                st.write(f"**Score de similarité:** {similarity_scores[idx]:.4f}")
                st.write("---")
        except Exception as e:
            st.error("Une erreur est survenue lors de la recherche.")
            st.write(f"Erreur : {e}")
