# app.py
# ──────────────────────────────────────────────────────────────
# Point d'entrée principal de l'application Streamlit.
# Ce fichier gère l'interface utilisateur, la logique de RAG (Retrieval-Augmented Generation),
# la persistance de l'historique des conversations et l'intégration du modèle Llama 3.1.
# ──────────────────────────────────────────────────────────────

import os
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# Importation des composants LangChain nécessaires pour le RAG
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import base64

# 1. Chargement des variables d'environnement
# Implication technique : La clé API (GROQ_API_KEY) est stockée dans un fichier .env 
# non versionné sur Git pour des raisons de sécurité. load_dotenv() la charge en mémoire.
load_dotenv()

# Chemin local des armoiries du Burkina Faso pour l'interface graphique
COAT_OF_ARMS_PATH = os.path.join(os.path.dirname(__file__), "assets", "armoiries_burkina_faso.png")

# Dossier de persistance de l'historique des conversations
# Choix technique : Stocker l'historique dans des fichiers JSON locaux (chat_history/).
# Implication : Permet une persistance simple et légère sans nécessiter de base de données relationnelle (SQL).
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "chat_history")
os.makedirs(HISTORY_DIR, exist_ok=True)

def get_base64_image(path):
    """
    Encode une image locale en base64.
    Choix technique : Streamlit ne gère pas toujours bien les images locales dans les balises HTML personnalisées.
    L'encodage en Base64 permet d'intégrer directement l'image dans le code HTML/CSS de la page,
    garantissant qu'elle s'affichera correctement à chaque rendu.
    """
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ── Fonctions de persistance de l'historique ──

def list_conversations():
    """
    Liste toutes les conversations sauvegardées.
    Choix technique : Lit chaque fichier JSON et extrait les métadonnées (titre, date).
    Implication : Le tri se fait en mémoire. Si le nombre de conversations devient immense, 
    cela pourrait ralentir l'app, mais pour un usage standard, c'est très performant.
    """
    convos = []
    for filename in os.listdir(HISTORY_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(HISTORY_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                convos.append({
                    "id": filename.replace(".json", ""),
                    "title": data.get("title", "Conversation sans titre"),
                    "date": data.get("date", ""),
                    "messages": data.get("messages", []),
                    "filepath": filepath
                })
            except (json.JSONDecodeError, IOError):
                # Ignore les fichiers corrompus pour ne pas faire planter l'application
                continue
    # Tri par date décroissante (les plus récents en premier)
    convos.sort(key=lambda c: c["date"], reverse=True)
    return convos

def save_conversation(conv_id, title, messages):
    """
    Sauvegarde l'état actuel de la conversation dans un fichier JSON.
    Implication technique : À chaque nouveau message, le fichier est écrasé avec le nouvel historique complet.
    C'est une opération rapide grâce à la petite taille des historiques textuels.
    """
    filepath = os.path.join(HISTORY_DIR, f"{conv_id}.json")
    data = {
        "title": title,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": messages
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def delete_conversation(conv_id):
    """Supprime physiquement le fichier JSON associé à un identifiant de conversation."""
    filepath = os.path.join(HISTORY_DIR, f"{conv_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)

def generate_conv_id():
    """Génère un identifiant unique (basé sur le timestamp) pour éviter les collisions de fichiers."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_title(messages):
    """
    Crée un titre dynamique pour la conversation en prenant les 50 premiers caractères
    de la première question posée par l'utilisateur. Cela améliore l'UX dans la barre latérale.
    """
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"][:50]
            return text + "..." if len(msg["content"]) > 50 else text
    return "Nouvelle conversation"

# Encodage de l'image dès le démarrage
COAT_OF_ARMS_B64 = get_base64_image(COAT_OF_ARMS_PATH)
COAT_OF_ARMS_URL = f"data:image/png;base64,{COAT_OF_ARMS_B64}"

# Configuration de la page Streamlit
# Choix technique : layout="centered" pour un design façon "Chatbot" centré et lisible,
# et sidebar="expanded" pour s'assurer que l'historique soit visible par défaut.
st.set_page_config(
    page_title="Assistant Juridique - Burkina Faso",
    page_icon="https://flagcdn.com/w40/bf.png",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
# CSS PREMIUM - Couleurs officielles du Burkina Faso
# ──────────────────────────────────────────────────────────────
# Choix technique : Externalisation du CSS complexe dans un fichier séparé (style.css).
# Implication : Rend le fichier app.py plus lisible et centralise le design.
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
with open(CSS_PATH, "r", encoding="utf-8") as f:
    st.markdown(f"<style>\n{f.read()}\n</style>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# SIDEBAR (Panneau latéral d'historique et options)
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    # Injection HTML/CSS pour afficher les armoiries et la devise dans la sidebar
    st.markdown(f"""
    <div class="sidebar-coa">
        <img src="{COAT_OF_ARMS_URL}" alt="Armoiries du Burkina Faso">
        <div class="sidebar-motto">La patrie ou la mort, nous vaincrons</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="flag-stripe"></div>', unsafe_allow_html=True)

    # ── Gestion des conversations ──
    st.markdown("### :speech_balloon: Conversations")

    # Bouton pour créer une nouvelle conversation. 
    # st.rerun() force le rafraîchissement complet de l'UI.
    if st.button("+ Nouvelle conversation", use_container_width=True, key="new_conv"):
        new_id = generate_conv_id()
        st.session_state.current_conv_id = new_id
        st.session_state.messages = []
        st.rerun()

    # Initialisation de la session (session_state est le cache mémoire spécifique au client)
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = generate_conv_id()

    # Affichage dynamique de l'historique des conversations
    conversations = list_conversations()
    if conversations:
        for convo in conversations:
            col_title, col_del = st.columns([5, 1])
            with col_title:
                is_active = (convo["id"] == st.session_state.get("current_conv_id"))
                label = f"{'> ' if is_active else ''}{convo['title']}"
                # Le bouton est grisé (disabled) s'il s'agit de la conversation actuellement active
                if st.button(
                    label,
                    key=f"conv_{convo['id']}",
                    use_container_width=True,
                    disabled=is_active
                ):
                    st.session_state.current_conv_id = convo["id"]
                    st.session_state.messages = convo["messages"]
                    st.rerun()
            with col_del:
                # Bouton de suppression
                if st.button("X", key=f"del_{convo['id']}"):
                    delete_conversation(convo["id"])
                    # Si on supprime la conversation en cours, on la remet à zéro
                    if st.session_state.get("current_conv_id") == convo["id"]:
                        st.session_state.current_conv_id = generate_conv_id()
                        st.session_state.messages = []
                    st.rerun()
    else:
        st.caption("Aucun historique pour le moment.")

    st.markdown('<div class="flag-stripe"></div>', unsafe_allow_html=True)

    # Accordéon informatif (st.expander)
    with st.expander("A propos & Fonctionnalites", expanded=False):
        st.markdown(
            "Cet assistant utilise l'**intelligence artificielle** (RAG + LLM) "
            "pour repondre a vos questions sur les demarches administratives "
            "au **Burkina Faso**."
        )
        st.markdown("""
<div class="feature-item">
    <span class="feature-icon">🔍</span>
    <span class="feature-text">Recherche intelligente dans les textes officiels</span>
</div>
<div class="feature-item">
    <span class="feature-icon">🛡️</span>
    <span class="feature-text">Anti-hallucination : reponses basees sur les sources</span>
</div>
<div class="feature-item">
    <span class="feature-icon">⚡</span>
    <span class="feature-text">Reponses instantanees via Groq + Llama 3.1</span>
</div>
<div class="feature-item">
    <span class="feature-icon">📚</span>
    <span class="feature-text">Base de connaissances extensible (.txt)</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("**Domaines couverts :**")
        st.markdown(
            "- Carte Nationale d'Identite (CNIB)\n"
            "- Passeport ordinaire\n"
            "- *Plus de procedures a venir...*"
        )

    st.markdown(
        "<div style='text-align:center; font-size:0.75rem; color: #5A6F94; padding: 0.5rem 0;'>"
        "Projet Master IFOAD &bull; 2026<br>"
        "Propulse par LangChain &amp; Groq"
        "</div>",
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────
# HERO HEADER avec armoiries
# ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-container">
    <div class="hero-coa">
        <img src="{COAT_OF_ARMS_URL}" alt="Armoiries du Burkina Faso">
    </div>
    <div class="hero-title">Assistant Juridique &amp; Administratif</div>
    <div class="hero-country">Burkina Faso</div>
    <div class="hero-subtitle">
        Votre guide intelligent pour les demarches administratives.
        Posez une question et obtenez une reponse precise basee sur les textes officiels.
    </div>
    <div class="hero-motto"><span>La patrie ou la mort</span> &bull; <span>nous vaincrons</span></div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# STAT CARDS avec couleurs du drapeau
# ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="stat-card red">
        <div class="stat-number">RAG</div>
        <div class="stat-label">Architecture</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-card gold">
        <div class="stat-number">Llama 3.1</div>
        <div class="stat-label">Modele IA</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-card green">
        <div class="stat-number">ChromaDB</div>
        <div class="stat-label">Base vectorielle</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# INITIALISATION DU RAG (Retrieval-Augmented Generation)
# ──────────────────────────────────────────────────────────────
# Choix technique : @st.cache_resource
# Implication : Empêche Streamlit de recharger le modèle d'embeddings, 
# la base de données ChromaDB et le client Groq à chaque interaction utilisateur. 
# Ces composants lourds sont conservés en mémoire cache globale.
@st.cache_resource
def initialiser_rag():
    # Chargement du modele d'embeddings
    # Utilisation de all-MiniLM-L6-v2 : Rapide, léger, idéal pour des recherches sémantiques courtes.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Connexion a la base vectorielle ChromaDB
    # persit_directory permet de lire la base persistée sur le disque (créée par l'ingestion).
    vector_db = Chroma(persist_directory="chromadb_storage", embedding_function=embeddings)

    # Configuration du "Retriever"
    # Choix technique : search_type="mmr" (Maximal Marginal Relevance) 
    # Implication : Au lieu de prendre juste les textes les plus similaires, MMR favorise la DIVERSITÉ 
    # pour éviter de ramener 5 fois le même paragraphe s'il est dupliqué.
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

    # Initialisation du LLM via l'API ultra-rapide de Groq (Llama 3.1)
    # temperature=0.0 : Choix CRITIQUE pour éliminer l'hallucination en forçant le modèle à être déterministe.
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=800
    )

    return retriever, llm

try:
    retriever, llm = initialiser_rag()

    # Prompt systeme strict
    # Choix technique : Directives en majuscules avec conditions d'échec claires.
    # Implication : "Prive" le LLM de sa capacité d'imagination pour le cantonner à un rôle de lecteur/extracteur de texte.
    system_prompt = (
        "Tu es un assistant juridique et administratif expert du Burkina Faso.\n"
        "1. Si l'utilisateur dit simplement bonjour, au revoir, merci, ou fait un petit commentaire, "
        "réponds simplement avec politesse sans utiliser le contexte.\n"
        "2. RÈGLE ABSOLUE ANTI-HALLUCINATION : Pour toute question sur les démarches, utilise EXCLUSIVEMENT les éléments de contexte fournis ci-dessous.\n"
        "3. N'INVENTE JAMAIS de noms d'institutions (comme OCTB), de prix, de délais, de pièces à fournir, ou d'étapes si ce n'est pas écrit noir sur blanc dans le contexte.\n"
        "4. Si tu ne connais pas la réponse ou si l'information n'est pas dans le contexte, TU DOIS RÉPONDRE EXACTEMENT : "
        "'Je ne dispose pas de cette information officielle dans ma base de connaissances actuelle.'\n"
        "5. Structure tes réponses de manière aérée et lisible.\n"
        "6. Ne cite JAMAIS plus de 10 éléments d'affilée dans une liste. Au-delà, utilise 'etc.'.\n"
        "7. Sois concis et NE RÉPÈTE JAMAIS LES MÊMES MOTS EN BOUCLE.\n"
        "Réponds toujours en français.\n\n"
        "Contexte :\n{context}"
    )

    # Création du Template de prompt LangChain (associe instructions système et question utilisateur)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Construction de la chaine de recuperation documentaire (Retrieval Chain)
    # Combine le LLM avec le Retriever. Le Retriever cherche les documents pertinents dans ChromaDB, 
    # et les insère dans la variable {context} du prompt avant d'appeler le LLM.
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # ── Gestion de l'historique de discussion local ──
    if "messages" not in st.session_state:
        # Tente de charger la conversation courante au démarrage de l'app ou suite à un rafraîchissement
        conv_id = st.session_state.get("current_conv_id", "")
        filepath = os.path.join(HISTORY_DIR, f"{conv_id}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                st.session_state.messages = data.get("messages", [])
            except (json.JSONDecodeError, IOError):
                st.session_state.messages = []
        else:
            st.session_state.messages = []

    # Message de bienvenue / Écran vide (Onboarding)
    if not st.session_state.messages:
        st.markdown(f"""
        <div class="welcome-card">
            <div class="welcome-coa">
                <img src="{COAT_OF_ARMS_URL}" alt="Armoiries">
            </div>
            <div class="welcome-title">Bienvenue ! Comment puis-je vous aider ?</div>
            <div class="welcome-text">
                Je suis votre assistant administratif virtuel au service des citoyens
                du Burkina Faso. Posez-moi une question sur les procedures officielles.
            </div>
            <div class="suggestions">
                <span class="chip">📋 Pieces pour la CNIB</span>
                <span class="chip">🛂 Demande de passeport</span>
                <span class="chip">💰 Cout du timbre</span>
                <span class="chip">🔄 Renouvellement CNIB</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Boucle d'affichage des messages existants (Le "Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Écoute de l'entrée utilisateur via le chat_input
    # Le code ci-dessous ne s'exécute QUE si l'utilisateur valide une question
    if user_query := st.chat_input("Posez votre question ici..."):
        # Afficher immédiatement la question dans l'interface
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Génération de la réponse via le système RAG
        with st.chat_message("assistant"):
            # st.spinner offre un feedback visuel pendant la latence réseau (appel API Groq)
            with st.spinner("Recherche dans les textes officiels..."):
                # Execution de la chaine LangChain
                response = rag_chain.invoke({"input": user_query})
                assistant_response = response["answer"]

                # Affichage de la réponse finale à l'écran
                st.markdown(assistant_response)

        # Enregistrement en RAM de la réponse
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        # Sauvegarde sur le disque (fichier JSON)
        conv_id = st.session_state.get("current_conv_id", generate_conv_id())
        title = extract_title(st.session_state.messages)
        save_conversation(conv_id, title, st.session_state.messages)

    # ──────────────────────────────────────────────────────────────
    # HACKS VISUELS (Auto-scroll)
    # ──────────────────────────────────────────────────────────────
    # Choix technique : Injection de code JavaScript via streamlit.components.
    # Implication : Streamlit par défaut ne défile pas toujours bien en bas après une très longue réponse. 
    # Ce petit code JS force le défilement de la div '.main' vers le bas pour une expérience plus fluide.
    import streamlit.components.v1 as components
    js_scroll = """
    <script>
        // Fonction pour scroller vers le bas
        function scrollToBottom() {
            var scrollNode = window.parent.document.querySelector('.main');
            if (scrollNode) {
                scrollNode.scrollTo({ top: scrollNode.scrollHeight, behavior: 'smooth' });
            }
        }
        // Exécuter après un petit délai pour s'assurer que le rendu est terminé
        setTimeout(scrollToBottom, 100);
    </script>
    """
    components.html(js_scroll, height=0)

    # Footer institutionnel
    st.markdown("""
    <div class="footer">
        <div class="footer-flag"></div>
        <div class="footer-motto">La patrie ou la mort, nous vaincrons</div>
        <div class="footer-text">Propulse par l'Intelligence Artificielle au service du citoyen</div>
        <div class="footer-badge">Burkina Faso &bull; Projet Master IFOAD 2026</div>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    # Gestion globale des erreurs au démarrage (ex: chromaDB manquant, clé API incorrecte)
    st.error(f"Erreur lors de l'initialisation de l'application : {e}")
    st.info("Verifiez que votre fichier .env contient la bonne cle et que vous avez execute l'ingestion.")