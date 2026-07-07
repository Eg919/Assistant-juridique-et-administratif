# super_scraper_rag.py
# ──────────────────────────────────────────────────────────────
# Script de Web Scraping avec Scrapy.
# Ce script visite les sites gouvernementaux du Burkina Faso, extrait le texte 
# visible des pages HTML, et télécharge/analyse les fichiers PDF correspondants
# aux mots-clés administratifs.
# ──────────────────────────────────────────────────────────────

import os
import time
import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from pypdf import PdfReader
from io import BytesIO

import urllib3
import sys
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# Choix technique : Désactivation des avertissements SSL (InsecureRequestWarning).
# Implication : Certains sites gouvernementaux peuvent avoir des certificats SSL expirés ou mal configurés.
# Cela permet au scraper de continuer à lire le site sans crasher pour des raisons de sécurité de certificat.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Forcer l'encodage UTF-8 dans la console pour l'affichage correct des caractères français
sys.stdout.reconfigure(encoding='utf-8')

# Configuration du chemin Tesseract sur Windows (pour l'OCR des PDF)
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# =======================================================================
# 🌐 TOUS LES SITES OFFICIELS DU BURKINA FASO À SCRAPER
# =======================================================================
# Choix technique : Liste exhaustive en dur (hardcodée).
# Implication : Le scraper ne se perdra pas sur le web mondial. Il reste cantonné 
# strictement aux domaines gouvernementaux (.bf ou .gov.bf) définis ici.
SITES_CIBLES_BF = [
    # Institutions Républicaines
    "https://www.presidence.bf",
    "https://www.gouvernement.gov.bf",
    "https://www.assembleenationale.bf",
    "https://conseil-constitutionnel.gov.bf",
    "https://www.jobf.gov.bf",
    
    # Démarches et Services Généraux
    "https://servicepublic.gov.bf",
    "https://oni.bf",
    "https://www.police.gov.bf",
    "https://www.me.bf",
    "https://www.impots.gov.bf",
    "https://www.douanes.bf",
    
    # Ministères (liste exhaustive pour extraire les lois et démarches)
    "https://www.fonction-publique.gov.bf",
    "https://www.justice.gov.bf",
    "https://www.finances.gov.bf",
    "https://www.sante.gov.bf",
    "https://www.education.gov.bf",
    "https://www.enseignementsuperieur.gov.bf",
    "https://www.defense.gov.bf",
    "https://www.securite.gov.bf",
    "https://www.maep.gov.bf",
    "https://www.environnement.gov.bf",
    "https://www.mines.gov.bf",
    "https://www.commerce.gov.bf",
    "https://www.infrastructures.gov.bf",
    "https://www.transports.gov.bf",
    "https://www.communication.gov.bf",
    "https://www.culture.gov.bf",
    "https://www.jeunesse.gov.bf",
    "https://www.sports.gov.bf",
    "https://www.action-sociale.gov.bf",
    "https://www.mfa.gov.bf"
]

# --- CONFIGURATION DES DOSSIERS DE STOCKAGE ---
DOSSIER_PDF = "data_scraped_pdf"
DOSSIER_TEXTE = "data"

for dossier in [DOSSIER_PDF, DOSSIER_TEXTE]:
    if not os.path.exists(dossier):
        os.makedirs(dossier)

# Filtres de mots-clés pour limiter le téléchargement de PDF
# Choix technique : On ne télécharge pas tous les PDF (car il peut y avoir des magazines, rapports, etc.).
# On filtre ceux dont le lien contient des mots-clés spécifiques (ex: constitution, loi, passeport).
MOTS_CLES = [
    "cnib", "passeport", "oni", "identification", "decret", "loi", "demarche", 
    "carte", "identite", "cefore", "constitution", "travail", "code", "statut", 
    "arrete", "reglement", "ordonnance", "circulaire", "guide", "formulaire"
]

class BurkinaSpider(scrapy.Spider):
    name = "burkina_spider"
    
    # Choix technique : Configuration fine de Scrapy (custom_settings)
    # Implication : 
    # - ROBOTSTXT_OBEY=False : Ignore le fichier robots.txt qui pourrait bloquer légitimement le scraping (attention à l'éthique).
    # - DEPTH_LIMIT=2 : Évite de scraper tout l'historique infini d'un site. Ne clique pas plus de 2 liens de profondeur.
    # - CLOSESPIDER_PAGECOUNT=150 : Sécurité absolue pour arrêter le scraper après 150 pages, évitant un plantage serveur ou un bannissement IP.
    # - CONCURRENT_REQUESTS=8 : Nombre de requêtes simultanées (plus rapide, mais charge le serveur distant).
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'DEPTH_LIMIT': 0, 
        'CLOSESPIDER_PAGECOUNT': 0, 
        'CONCURRENT_REQUESTS': 8,
        'DOWNLOAD_DELAY': 0.5, # Pause de 0.5s entre chaque requête pour être "poli" envers le serveur
        'LOG_LEVEL': 'DEBUG' 
    }
    
    start_urls = SITES_CIBLES_BF
    
    def parse(self, response):
        """
        Fonction appelée automatiquement pour chaque URL visitée par Scrapy.
        """
        # Ne traiter que les pages qui sont du texte/html (ignore les flux vidéo, images non ciblées)
        if not hasattr(response, 'text'):
            return

        # Extraction du domaine (ex: www.gouvernement.gov.bf) pour nommer le fichier texte cible
        domain = urlparse(response.url).netloc
        safe_domain = "".join(x for x in domain if x.isalnum() or x in "._-")
        chemin_txt_site = os.path.join(DOSSIER_TEXTE, f"site_{safe_domain}.txt")
        
        # Initialisation du fichier de stockage par domaine à la première écriture
        # Choix technique : Utilisation d'un set (self.cleared_files)
        # Implication : Assure qu'on écrase (w) le fichier texte du domaine la toute première fois
        # qu'on le visite, puis on ajoute (a) le contenu des pages suivantes à la suite.
        if not hasattr(self, 'cleared_files'):
            self.cleared_files = set()
            
        if chemin_txt_site not in self.cleared_files:
            with open(chemin_txt_site, "w", encoding="utf-8") as txt_file:
                txt_file.write(f"=== COMPILATION DU SITE : {domain} ===\n\n")
            self.cleared_files.add(chemin_txt_site)

        # --- Extraction intelligente du texte HTML ---
        # Choix technique : Utilisation de BeautifulSoup
        # Implication : Permet de supprimer proprement le code inutile (javascript, CSS, menus, footers)
        # pour ne garder que le contenu "texte" sémantique de la page web. Cela donne une donnée RAG très propre.
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.extract() # Supprime ces balises de l'arbre HTML
            
        texte_page = soup.get_text(separator='\n', strip=True)
        
        if texte_page:
            # Ajout du texte extrait au fichier global du domaine
            with open(chemin_txt_site, "a", encoding="utf-8") as txt_file:
                txt_file.write(f"\n\n--- PAGE : {response.url} ---\n")
                txt_file.write(texte_page)

        # --- Extraction et suivi des liens ---
        # Scrapy récupère tous les balises <a> (liens)
        links = response.css('a::attr(href)').getall()
        texts = response.css('a::text').getall()

        for href, text_lien in zip(links, texts):
            url_complete = response.urljoin(href)
            text_lien_lower = (text_lien or '').lower()
            href_lower = href.lower()
            
            # Vérifier si c'est un fichier PDF
            if ".pdf" in url_complete.lower():
                # On télécharge uniquement si le texte du lien ou l'URL contient un des MOTS_CLES
                if any(mot in href_lower or mot in text_lien_lower for mot in MOTS_CLES) or "demarche" in href_lower or "download" in href_lower:
                    yield scrapy.Request(
                        url_complete, 
                        callback=self.save_pdf, 
                        meta={'text_lien': text_lien_lower} # Passe le texte du lien à la fonction save_pdf
                    )
            # Sinon, suivre le lien s'il appartient au même domaine (pour ne pas sortir des sites BF)
            # Correction: on enlève le 'www.' éventuel pour ne pas ignorer les pages internes
            elif urlparse(url_complete).netloc.replace("www.", "") == domain.replace("www.", ""):
                # Éviter les images ou archives volumineuses inutiles
                if not any(url_complete.lower().endswith(ext) for ext in [".jpg", ".png", ".zip", ".doc", ".docx", ".xls", ".xlsx"]):
                    # Ordonne à Scrapy de visiter ce nouveau lien et d'appeler à nouveau la fonction parse
                    yield response.follow(url_complete, callback=self.parse)

    def save_pdf(self, response):
        """
        Callback exécuté uniquement lorsque Scrapy télécharge un fichier PDF.
        Gère le téléchargement, la lecture native, et le fallback vers OCR si nécessaire.
        """
        # Utilisation du texte du lien cliqué pour donner un nom de fichier lisible au PDF
        text_lien = response.meta.get('text_lien', '')
        nom_propre = "".join(x for x in (text_lien or 'document') if x.isalnum() or x in "._- ").strip().replace(" ", "_")[:50]
        if not nom_propre:
            nom_propre = f"doc_{int(time.time())}" # Fallback si le texte du lien est vide
            
        chemin_pdf = os.path.join(DOSSIER_PDF, f"{nom_propre}.pdf")
        chemin_txt = os.path.join(DOSSIER_TEXTE, f"{nom_propre}.txt")

        self.logger.info(f"📥 Téléchargement du PDF réussi : {response.url}")
        
        # Sauvegarde du PDF brut sur le disque
        with open(chemin_pdf, "wb") as f:
            f.write(response.body)

        # Extraction du texte du PDF
        try:
            # Pypdf extrait le texte numérique s'il existe
            reader = PdfReader(BytesIO(response.body))
            texte_integral = ""
            for i, page in enumerate(reader.pages):
                texte_page = page.extract_text()
                if texte_page:
                    texte_integral += f"\n--- Page {i+1} ---\n" + texte_page
            
            # OCR (Optical Character Recognition) si le texte extrait est inférieur à 50 caractères
            # Choix technique : Intégration de Tesseract au sein même du workflow Scrapy.
            # Implication : Scrapy est asynchrone, mais Tesseract est synchrone et très lourd.
            # Cela pourrait ralentir l'araignée (Spider), mais garantit qu'aucun document scanné n'est perdu.
            if len(texte_integral.strip()) < 50:
                self.logger.info(f"⚠️ Texte vide ou insuffisant détecté pour {nom_propre}. Lancement de l'OCR...")
                texte_integral = ""
                doc = fitz.open(stream=response.body, filetype="pdf")
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=200) # Résolution pour l'OCR
                    img_bytes = pix.tobytes("png")
                    img = Image.open(BytesIO(img_bytes))
                    texte_ocr = pytesseract.image_to_string(img, lang="fra+eng")
                    if texte_ocr.strip():
                        texte_integral += f"\n--- Page {page_num+1} (OCR) ---\n" + texte_ocr
                doc.close()

            # Enregistrement du texte final dans le dossier "data/" pour qu'il soit indexé plus tard par ChromaDB
            if texte_integral.strip():
                with open(chemin_txt, "w", encoding="utf-8") as txt_file:
                    txt_file.write(f"---\nTITRE : {nom_propre.replace('_', ' ')}\nSOURCE : {response.url}\n---\n\n")
                    txt_file.write(texte_integral)
                self.logger.info(f"✅ Texte extrait du PDF avec succès : {chemin_txt}")
            else:
                self.logger.warning(f"⚠️ Document PDF scanné sans OCR (texte vide) : {chemin_pdf}")
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du traitement du document PDF {response.url} : {e}")

if __name__ == "__main__":
    print("--- 🚀 DÉBUT DU SCRAPING GLOBAL (AVEC SCRAPY) ---")
    
    # Lancement du processus Scrapy en mode script (sans utiliser la CLI 'scrapy crawl')
    process = CrawlerProcess()
    process.crawl(BurkinaSpider)
    process.start()
    
    print("--- ✅ SCRAPING TERMINÉ ---")