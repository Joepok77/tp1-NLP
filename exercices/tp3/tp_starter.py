"""
TP Seance 3 - Classification par prompt engineering (OpenAI / Gemini)
AI2 - M2 Data Scientist & IA - M3 NLP & Transformers

Completez les sections marquees TODO, dans l'ordre des etapes du TP.
Necessite une cle d'API valide (OPENAI_API_KEY ou GEMINI_API_KEY) en variable
d'environnement -- voir README.md.
"""

import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv() 

DATA_PATH = "tickets_support_iot_v2.csv"
CATEGORIES = ["connectivite", "batterie", "firmware", "securite", "installation", "interface"]

# ---------------------------------------------------------------------------
# Etape 1 - Prise en main
# ---------------------------------------------------------------------------

def appeler_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Envoie un prompt au LLM et retourne le texte de la reponse.

    TODO :
      - instancier le client OpenAI (ou Gemini)
      - envoyer le prompt avec temperature=0.0 (deterministe pour la classification)
      - retourner le texte de la reponse
    """
    client = genai.Client()  # connection avec la cle GEMINI_API_KEY 
    reponse = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    print("  tokens consommes :", reponse.usage_metadata.total_token_count)
    return reponse.text.strip()


# ---------------------------------------------------------------------------
# Etape 2 - Classification zero-shot
# ---------------------------------------------------------------------------

def prompt_zero_shot(texte: str) -> str:
    """Construit un prompt zero-shot demandant une classification stricte.

    TODO : retourner une chaine qui liste les categories possibles et
    demande une reponse au format "un seul mot, la categorie exacte".
    """
    raise NotImplementedError


def nettoyer_reponse(reponse: str) -> str:
    """Normalise la reponse du modele pour la comparer aux categories connues.

    TODO :
      - mettre en minuscules, retirer les espaces/ponctuation superflus
      - chercher quelle categorie de CATEGORIES apparait dans la reponse
      - retourner 'INCONNUE' si aucune categorie n'est reconnue
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Etape 3 - Classification few-shot
# ---------------------------------------------------------------------------

def prompt_few_shot(texte: str, exemples: list) -> str:
    """Construit un prompt few-shot a partir d'une liste d'exemples (texte, categorie).

    TODO : retourner un prompt qui montre les exemples puis demande la
    classification du nouveau texte, dans le meme format que zero-shot.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Etape 4 - Chain-of-thought
# ---------------------------------------------------------------------------

def prompt_cot(texte: str) -> str:
    """Construit un prompt qui demande un raisonnement explicite avant la reponse.

    TODO : demander un raisonnement etape par etape, puis exiger une
    conclusion au format exact "Categorie: <nom>".
    """
    raise NotImplementedError


def extraire_categorie_cot(reponse: str) -> str:
    """Extrait la categorie finale d'une reponse chain-of-thought.

    TODO : utiliser une expression reguliere pour chercher "Categorie: <nom>"
    dans le texte de la reponse, plutot qu'une simple recherche de sous-chaine.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Etape 5 - Comparatif final
# ---------------------------------------------------------------------------

def evaluer_approche(df_echantillon: pd.DataFrame, fonction_prompt, fonction_extraction, pause: float = 0.5):
    """Applique une approche de classification a un echantillon et calcule l'accuracy.

    TODO :
      - pour chaque ticket : construire le prompt, appeler le LLM, extraire la categorie
      - comparer a la vraie categorie (colonne 'categorie')
      - inserer un temps de pause (time.sleep(pause)) entre chaque appel
      - retourner l'accuracy (float) et la liste des predictions
    """
    raise NotImplementedError


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    echantillon = df.sample(n=15, random_state=42)
    print("Echantillon de test :", len(echantillon), "tickets")

    # Etape 1 - premier appel au LLM
    print(appeler_llm("Explique en une phrase ce qu'est un capteur IoT", model="gemini-3.5-flash-lite"))

    # Decommentez au fur et a mesure de votre progression :
    # acc_zero, preds_zero = evaluer_approche(echantillon, prompt_zero_shot, nettoyer_reponse)
    # print("Accuracy zero-shot :", acc_zero)
