# database_ingestion.py
# ──────────────────────────────────────────────────────────────
# Script d'ingestion de la base de connaissances.
# Ce script prend les fichiers texte (.txt) du dossier "data/",
# les découpe en petits morceaux (chunks) et les vectorise avec un modèle
# d'embeddings avant de les stocker dans la base vectorielle ChromaDB.
# ──────────────────────────────────────────────────────────────

import os
import sys

# Importation des outils LangChain pour le traitement documentaire
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Choix technique : Forcer l'encodage UTF-8 pour la sortie standard.
# Implication : Évite les plantages sur les terminaux Windows qui utilisent souvent cp1252 
# lorsqu'ils essaient d'imprimer des émojis ou des caractères accentués (français).
sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("--- 🚀 Début du processus d'ingestion RAG ---")
    
    # 1. Vérification et chargement du dossier de données
    # Choix technique : Création automatique du dossier "data" s'il n'existe pas.
    # Implication : Prévient les erreurs de type "FileNotFoundError" lors de la première exécution.
    if not os.path.exists("data"):
        os.makedirs("data")
        print("📁 Dossier 'data/' créé. Placez-y vos fichiers .txt administratifs.")
        return

    print("📖 Chargement des documents en cours...")
    
    # Choix technique : Utilisation de DirectoryLoader avec TextLoader.
    # Implication : Charge en mémoire tous les fichiers .txt d'un coup. C'est rapide mais 
    # cela suppose que la base documentaire tienne dans la RAM de la machine (ce qui est le cas pour du texte).
    loader = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    
    if not documents:
        print("⚠️ Aucun fichier texte trouvé dans le dossier 'data/'.")
        return
    print(f"✅ {len(documents)} document(s) chargé(s) avec succès.")

    # 2. Découpage des textes (Chunking stratégique)
    # Choix technique : RecursiveCharacterTextSplitter.
    # Implication : Il tente d'abord de couper au niveau des paragraphes (\n\n), 
    # puis des phrases, puis des mots pour ne pas couper le sens d'une phrase en plein milieu.
    # Les paramètres (chunk_size=500, chunk_overlap=50) assurent que chaque morceau est assez court 
    # pour que l'IA le lise vite, avec 50 caractères de chevauchement pour ne pas perdre le contexte 
    # entre deux morceaux adjacents.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Textes segmentés en {len(chunks)} morceaux (chunks).")

    # 3. Initialisation du modèle d'Embeddings
    # Choix technique : HuggingFaceEmbeddings ("sentence-transformers/all-MiniLM-L6-v2")
    # Implication : C'est un modèle open-source, gratuit, qui s'exécute localement sans clé API.
    # Il est très léger (optimisé L6) et permet de transformer le texte en vecteurs mathématiques rapidement.
    print("🧠 Initialisation du modèle d'embeddings (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Création et persistance de la base de données vectorielle ChromaDB
    # Choix technique : Stockage local avec persist_directory="chromadb_storage".
    # Implication : Au lieu d'utiliser une base vectorielle payante dans le cloud (comme Pinecone),
    # ChromaDB enregistre l'index directement sur le disque. C'est gratuit, privé, et l'application Streamlit 
    # n'aura qu'à lire ce dossier au démarrage sans refaire l'ingestion.
    print("💾 Stockage des vecteurs dans la base locale ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chromadb_storage" # Ce dossier contiendra les index vectoriels (.sqlite3, etc.)
    )
    
    print("✨ --- Étape 1 Réussie : Base vectorielle créée avec succès ! ---")

if __name__ == "__main__":
    main()