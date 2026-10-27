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
MODELE = "gemini-3.5-flash-lite"

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
    return (
        "Classe ce ticket de support client dans une seule de ces catégories : "
        + ", ".join(CATEGORIES) + ".\n"
        "Réponds uniquement par le nom exact de la catégorie, en un seul mot, sans rien d'autre.\n\n"
        "Ticket : " + texte
    )


def nettoyer_reponse(reponse: str) -> str:
    """Normalise la reponse du modele pour la comparer aux categories connues.

    TODO :
      - mettre en minuscules, retirer les espaces/ponctuation superflus
      - chercher quelle categorie de CATEGORIES apparait dans la reponse
      - retourner 'INCONNUE' si aucune categorie n'est reconnue
    """
    reponse = reponse.lower().replace("é", "e").replace("è", "e")  # "Sécurité" -> "securite"
    mots = re.sub(r"[^a-z]", " ", reponse).split()                  # retire ponctuation, ** etc.
    for categorie in CATEGORIES:
        if categorie in mots:
            return categorie
    return "INCONNUE"


# ---------------------------------------------------------------------------
# Etape 3 - Classification few-shot
# ---------------------------------------------------------------------------

def prompt_few_shot(texte: str, exemples: list) -> str:
    """Construit un prompt few-shot a partir d'une liste d'exemples (texte, categorie).

    TODO : retourner un prompt qui montre les exemples puis demande la
    classification du nouveau texte, dans le meme format que zero-shot.
    """
    prompt = (
        "Classe ce ticket de support client dans une seule de ces catégories : "
        + ", ".join(CATEGORIES) + ".\n"
        "Réponds uniquement par le nom exact de la catégorie, en un seul mot, sans rien d'autre.\n\n"
        "Exemples :\n"
    )
    for texte_exemple, categorie in exemples:
        prompt += f"Ticket : {texte_exemple}\nCatégorie : {categorie}\n\n"
    prompt += f"Ticket : {texte}\nCatégorie :"
    return prompt


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
    predictions = []
    bonnes = 0
    for _, ligne in df_echantillon.iterrows():
        reponse = appeler_llm(fonction_prompt(ligne["texte"]), model=MODELE)
        prediction = fonction_extraction(reponse)
        predictions.append(prediction)
        if prediction == ligne["categorie"]:
            bonnes += 1
        print(f"  vraie: {ligne['categorie']:<13} predite: {prediction:<13} brute: {reponse!r}")
        time.sleep(pause)
    return bonnes / len(df_echantillon), predictions


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    echantillon = df.sample(n=15, random_state=42)
    print("Echantillon de test :", len(echantillon), "tickets")

    # Etape 1 - premier appel au LLM
    print(appeler_llm("Explique en une phrase ce qu'est un capteur IoT", model="gemini-3.5-flash-lite"))

    # Decommentez au fur et a mesure de votre progression :
    acc_zero, preds_zero = evaluer_approche(echantillon, prompt_zero_shot, nettoyer_reponse)
    print("Accuracy zero-shot :", acc_zero)

    # Etape 3 - few-shot : 1 exemple par categorie, choisis HORS des 15 tickets de test
    exemples = [
        ("La batterie du module Edge se décharge beaucoup trop vite, moins de deux jours d'autonomie contre sept annoncés.", "batterie"),
        ("Après la mise à jour OTA du firmware version 3.2, la passerelle redémarre en boucle toutes les dix minutes.", "firmware"),
        ("Impossible d'appairer le nouveau capteur de vibration avec le broker MQTT central, erreur de timeout systématique.", "connectivite"),
        ("La documentation d'installation du kit Edge AI ne mentionne pas la configuration du VPN, nous sommes bloqués.", "installation"),
        ("L'application mobile ne permet toujours pas d'exporter les rapports de supervision en PDF, fonctionnalité pourtant annoncée.", "interface"),
        ("Nous avons détecté une tentative de connexion suspecte sur un de nos capteurs OPC UA la nuit dernière.", "securite"),
    ]

    # evaluer_approche() donne seulement le texte a la fonction de prompt : on lui ajoute les exemples
    def prompt_few_shot_tp(texte):
        return prompt_few_shot(texte, exemples)

    acc_few, preds_few = evaluer_approche(echantillon, prompt_few_shot_tp, nettoyer_reponse, pause=4)
    print("Accuracy few-shot :", acc_few)
