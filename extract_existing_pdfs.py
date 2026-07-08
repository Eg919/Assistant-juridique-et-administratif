# extract_existing_pdfs.py
# ──────────────────────────────────────────────────────────────
# Script utilitaire d'extraction de texte depuis des fichiers PDF.
# Si un PDF est composé d'images scannées (sans texte sélectionnable),
# le script utilise la reconnaissance optique de caractères (OCR) via Tesseract.
# ──────────────────────────────────────────────────────────────

import os
import sys
from io import BytesIO

# pypdf : Bibliothèque native Python pour lire le texte directement encapsulé dans les PDF
from pypdf import PdfReader

# PyMuPDF (fitz) : Utilisé pour extraire les pages du PDF sous forme d'images haute résolution
import fitz  

# pytesseract & PIL : Utilisés pour l'OCR (Reconnaissance Optique de Caractères) sur les images
import pytesseract
from PIL import Image

# Choix technique : Forcer l'encodage UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Configuration du chemin Tesseract sur Windows
# Implication technique : pytesseract n'est qu'une surcouche Python. Il a besoin du binaire exécutable
# Tesseract-OCR installé sur la machine hôte pour fonctionner. Ce chemin doit correspondre à l'installation locale.
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    print(f"ATTENTION: Tesseract introuvable à {tesseract_path}. L'OCR risque d'échouer.")

# Définition des répertoires source (PDF) et destination (TXT)
DOSSIER_PDF = "data_scraped_pdf"
DOSSIER_TEXTE = "data"

if not os.path.exists(DOSSIER_TEXTE):
    os.makedirs(DOSSIER_TEXTE)

def extract_pdfs():
    """
    Parcourt le dossier contenant les PDF et extrait leur texte.
    Gère intelligemment le repli (fallback) sur l'OCR si le texte natif est vide.
    """
    print("--- Extraction des PDFs existants ---")
    if not os.path.exists(DOSSIER_PDF):
        print(f"Le dossier {DOSSIER_PDF} n'existe pas.")
        return
        
    for fichier in os.listdir(DOSSIER_PDF):
        if fichier.lower().endswith(".pdf"):
            chemin_pdf = os.path.join(DOSSIER_PDF, fichier)
            nom_propre = fichier[:-4]
            chemin_txt = os.path.join(DOSSIER_TEXTE, f"{nom_propre}.txt")
            
            try:
                # Étape 1 : Tentative d'extraction native avec pypdf
                # Choix technique : pypdf est très rapide et consomme peu de CPU/RAM.
                reader = PdfReader(chemin_pdf)
                texte_integral = ""
                for i, page in enumerate(reader.pages):
                    texte_page = page.extract_text()
                    if texte_page:
                        texte_integral += f"\n--- Page {i+1} ---\n" + texte_page
                
                # Étape 2 : Mécanisme de Fallback vers l'OCR
                # Choix technique : Si le texte extrait est inférieur à 50 caractères, on suppose 
                # qu'il s'agit d'un scan ou d'un document image.
                # Implication : Évite de passer à la moulinette OCR des PDF normaux (car l'OCR est très lent 
                # et gourmand en ressources), mais sauve les documents purement administratifs "scannés".
                if len(texte_integral.strip()) < 50:
                    print(f"⚠️ Texte vide ou insuffisant détecté pour {nom_propre}. Lancement de l'OCR...")
                    texte_integral = ""
                    doc = fitz.open(chemin_pdf)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        
                        # Choix technique : Résolution de 200 DPI pour le rendu en image.
                        # Implication : C'est le "sweet spot" pour Tesseract : assez net pour bien lire les caractères,
                        # mais pas trop énorme pour éviter de saturer la RAM.
                        pix = page.get_pixmap(dpi=200) 
                        img_bytes = pix.tobytes("png")
                        img = Image.open(BytesIO(img_bytes))
                        
                        # Choix technique : Langues "fra+eng"
                        # Implication : Tesseract tentera de reconnaître le français et l'anglais, améliorant
                        # la gestion des termes techniques ou acronymes.
                        texte_ocr = pytesseract.image_to_string(img, lang="fra+eng")
                        
                        if texte_ocr.strip():
                            texte_integral += f"\n--- Page {page_num+1} (OCR) ---\n" + texte_ocr
                    doc.close()
                
                # Étape 3 : Sauvegarde du résultat
                if texte_integral.strip():
                    # Choix technique : Ajout d'un en-tête (métadonnées simples) au début du fichier texte.
                    # Implication : Lors du découpage (chunking) par l'ingestion, le premier chunk 
                    # retiendra le titre du document, ce qui aide grandement la recherche vectorielle (RAG).
                    with open(chemin_txt, "w", encoding="utf-8") as txt_file:
                        txt_file.write(f"---\nTITRE : {nom_propre.replace('_', ' ')}\nSOURCE : local\n---\n\n")
                        txt_file.write(texte_integral)
                    print(f"✅ Texte extrait du PDF avec succès : {chemin_txt}")
                else:
                    print(f"⚠️ Document PDF scanné sans OCR (texte vide) : {chemin_pdf}")
                    
            except Exception as e:
                # Capture d'erreur robuste pour qu'un PDF corrompu n'arrête pas la boucle sur les 100 autres
                print(f"❌ Erreur lors du traitement du document PDF {chemin_pdf} : {e}")

if __name__ == "__main__":
    extract_pdfs()
