# Assistant Juridique & Administratif (Version Streamlit) - Burkina Faso 🇧🇫

Ce projet est un Agent IA basé sur une architecture RAG (Retrieval-Augmented Generation). Il a pour but d'aider les citoyens burkinabè à s'informer sur leurs démarches administratives (CNIB, Passeport, CEFORE, etc.) en se basant strictement sur des textes officiels.

## 🛠️ Prérequis

Assurez-vous d'avoir installé Python **3.11** (la version exacte utilisée lors du développement de ce projet est Python 3.11.9).

## 🚀 Étapes de démarrage

Suivez ces instructions dans l'ordre pour configurer et lancer le projet depuis zéro.

### 1. Cloner ou télécharger le projet
Placez-vous dans le dossier racine du projet avec votre terminal.

### 2. Créer et activer l'environnement virtuel
Il est fortement recommandé de travailler dans un environnement virtuel pour isoler les dépendances.

**Sur Windows :**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Sur macOS/Linux :**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```powershell
pip install -r requirements.txt
```

### 4. Variables d'environnement
Assurez-vous que le fichier `.env` est présent et contient votre clé API Groq :
`GROQ_API_KEY=votre_cle_ici`

### 5. Scraping et Ingestion de la base de données
Avant de lancer l'application pour la première fois (ou si vous souhaitez mettre à jour les données), vous devez extraire les textes des sites officiels et les indexer :

1. Lancez le scraper pour récupérer les données (cela va remplir le dossier `data/`) :
```powershell
python super_scraper_rag.py
```
*(Astuce : Le scraper peut prendre du temps. Vous pouvez modifier `CLOSESPIDER_PAGECOUNT` dans le script pour limiter la profondeur).*

2. Ingérez les données texte dans la base vectorielle ChromaDB :
```powershell
python database_ingestion.py
```

### 6. Lancement de l'application (Streamlit)
```powershell
streamlit run app.py
```
L'interface web s'ouvrira automatiquement dans votre navigateur par défaut.
