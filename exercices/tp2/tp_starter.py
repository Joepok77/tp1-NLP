"""
TP Seance 1 - Pipeline NLP complet avec spaCy et NLTK
AI2 - M2 Data Scientist & IA - M3 NLP & Transformers

Completez les fonctions marquees TODO, dans l'ordre des etapes du TP.
Executez ce fichier section par section (ou copiez les fonctions dans un notebook).
"""

import re
from pathlib import Path
import pandas as pd
import spacy
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Chemin du CSV a cote de ce script, pour pouvoir lancer le script depuis n'importe quel dossier
DATA_PATH = Path(__file__).parent / "tickets_support_iot.csv"

# ---------------------------------------------------------------------------
# Etape 1 - Prise en main
# ---------------------------------------------------------------------------

def charger_donnees(path: str) -> pd.DataFrame:
    """Charge le CSV de tickets et retourne un DataFrame pandas.

    TODO :
      - lire le fichier CSV avec pandas
      - verifier qu'il contient bien les colonnes id, categorie, texte
      - retourner le DataFrame
    """
    df = pd.read_csv(path)

    colonnes_attendues = {"id", "categorie", "texte"}
    manquantes = colonnes_attendues - set(df.columns)
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans le CSV : {manquantes}")

    return df


# ---------------------------------------------------------------------------
# Etape 2 - Nettoyage et tokenization (guidee)
# ---------------------------------------------------------------------------

def nettoyer_texte(texte: str) -> str:
    """Nettoie un texte brut : minuscules, suppression de la ponctuation superflue.

    TODO :
      - mettre le texte en minuscules
      - supprimer les caracteres qui ne sont ni des lettres, ni des espaces,
        ni des apostrophes/tirets internes aux mots
      - reduire les espaces multiples a un seul espace
    """
    texte = texte.lower()
    # remplace par un espace tout ce qui n'est ni lettre, ni espace, ni ' ou - (chiffres et _ compris)
    texte = re.sub(r"[^\w\s'-]|[\d_]", " ", texte)
    # ne garde ' et - qu'entre deux lettres (l'icone, wi-fi), pas isoles ou en bord de mot
    texte = re.sub(r"(?<!\w)['-]|['-](?!\w)", " ", texte)
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()


def tokenizer_nltk(texte_nettoye: str) -> list:
    """Tokenize un texte nettoye avec NLTK (word_tokenize)."""
    return word_tokenize(texte_nettoye, language="french")


def tokenizer_spacy(texte_nettoye: str, nlp) -> list:
    """Tokenize un texte nettoye avec spaCy et retourne la liste des tokens (texte brut)."""
    return [token.text for token in nlp(texte_nettoye)]


# ---------------------------------------------------------------------------
# Etape 3 - Stopwords, lemmatisation, POS tagging (approfondissement)
# ---------------------------------------------------------------------------

def supprimer_stopwords(tokens: list, stopwords_fr: set) -> list:
    """Retire les mots vides (stopwords) d'une liste de tokens."""
    # la liste NLTK contient "l", "d"... sans apostrophe, alors que spaCy produit "l'", "d'"
    return [token for token in tokens if token.rstrip("'") not in stopwords_fr]


def lemmatiser_spacy(texte_nettoye: str, nlp) -> list:
    """Retourne la liste des lemmes (formes canoniques) des tokens du texte, via spaCy."""
    return [token.lemma_ for token in nlp(texte_nettoye)]


def filtrer_par_pos(texte_nettoye: str, nlp, pos_autorises=("NOUN", "VERB", "ADJ")) -> list:
    """Ne garde que les lemmes dont l'etiquette POS (part-of-speech) est dans pos_autorises."""
    return [token.lemma_ for token in nlp(texte_nettoye) if token.pos_ in pos_autorises]


# ---------------------------------------------------------------------------
# Etape 4 - Entites nommees et analyse de frequence (autonomie)
# ---------------------------------------------------------------------------

def extraire_entites(texte: str, nlp) -> list:
    """Retourne la liste des entites nommees (texte, label) detectees par spaCy."""
    # sur le texte brut : les majuscules aident spaCy a reperer les noms propres
    return [(ent.text, ent.label_) for ent in nlp(texte).ents]


def vocabulaire_frequent(tous_les_tokens: list, top_n: int = 20) -> list:
    """Retourne les top_n tokens les plus frequents (token, occurrences)."""
    return Counter(tous_les_tokens).most_common(top_n)


# ---------------------------------------------------------------------------
# Etape 5 - Pipeline complet et rapport (challenge final)
# ---------------------------------------------------------------------------

def pipeline_complet(texte: str, nlp, stopwords_fr: set) -> list:
    """Enchaine nettoyage -> tokenization -> stopwords -> lemmatisation -> filtre POS.

    Doit retourner la liste finale de lemmes exploitables pour un ticket.
    """
    raise NotImplementedError


def rapport_par_categorie(df: pd.DataFrame, nlp, stopwords_fr: set) -> pd.DataFrame:
    """Pour chaque categorie de ticket, calcule :
      - le nombre de tickets
      - la taille du vocabulaire brut (avant nettoyage)
      - la taille du vocabulaire apres pipeline_complet
      - les 5 lemmes les plus frequents

    Retourne un DataFrame recapitulatif, une ligne par categorie.
    """
    raise NotImplementedError


if __name__ == "__main__":
    nlp = spacy.load("fr_core_news_sm")
    stop_fr = set(stopwords.words("french"))

    # Etape 1
    df = charger_donnees(DATA_PATH)
    print(df.head())
    print("\nNombre total de tickets :", len(df))
    print("\nRepartition par categorie :")
    print(df["categorie"].value_counts())

    # Etape 2 - comparaison NLTK / spaCy sur le premier ticket
    exemple = df.loc[0, "texte"]
    nettoye = nettoyer_texte(exemple)
    tokens_nltk = tokenizer_nltk(nettoye)
    tokens_spacy = tokenizer_spacy(nettoye, nlp)

    print("\nTexte brut    :", exemple)
    print("Texte nettoye :", nettoye)
    print(f"\nNLTK  ({len(tokens_nltk)} tokens) :", tokens_nltk)
    print(f"spaCy ({len(tokens_spacy)} tokens) :", tokens_spacy)
    print("Seulement NLTK  :", sorted(set(tokens_nltk) - set(tokens_spacy)))
    print("Seulement spaCy :", sorted(set(tokens_spacy) - set(tokens_nltk)))

    # Meme comparaison sur tout le corpus
    textes_nettoyes = df["texte"].apply(nettoyer_texte)
    total_nltk = sum(len(tokenizer_nltk(t)) for t in textes_nettoyes)
    total_spacy = sum(len(tokenizer_spacy(t, nlp)) for t in textes_nettoyes)
    print(f"\nSur les {len(df)} tickets : {total_nltk} tokens NLTK, {total_spacy} tokens spaCy")

    # Etape 3 - tokens bruts -> sans stopwords -> lemmes filtres par POS (premier ticket)
    sans_stopwords = supprimer_stopwords(tokens_spacy, stop_fr)
    lemmes = lemmatiser_spacy(nettoye, nlp)
    lemmes_pos = filtrer_par_pos(nettoye, nlp)

    print(f"\nTokens bruts         ({len(tokens_spacy)}) :", tokens_spacy)
    print(f"Sans stopwords       ({len(sans_stopwords)}) :", sans_stopwords)
    print(f"Lemmes               ({len(lemmes)}) :", lemmes)
    print(f"Lemmes filtres (POS) ({len(lemmes_pos)}) :", lemmes_pos)
    print("Detail POS :", [(token.text, token.pos_) for token in nlp(nettoye)])

    # Etape 4 - entites nommees sur une phrase avec un lieu et une organisation
    phrase = "Le technicien de Schneider Electric est intervenu sur le site de Lyon pour configurer le broker MQTT."
    print("\nPhrase test :", phrase)
    print("Entites :", extraire_entites(phrase, nlp))

    # Entites detectees dans tout le corpus (texte brut)
    entites_corpus = Counter(e for t in df["texte"] for e in extraire_entites(t, nlp))
    print("\nEntites du corpus :", entites_corpus.most_common())

    # 15 lemmes les plus frequents apres le pipeline de l'etape 3
    tous_les_lemmes = []
    for t in textes_nettoyes:
        tous_les_lemmes += supprimer_stopwords(filtrer_par_pos(t, nlp), stop_fr)
    print("\nTop 15 lemmes :", vocabulaire_frequent(tous_les_lemmes, 15))

    # Decommentez au fur et a mesure de votre progression dans le TP :
    # exemple = df.loc[0, "texte"]
    # print(nettoyer_texte(exemple))
    # print(pipeline_complet(exemple, nlp, stop_fr))
    # print(rapport_par_categorie(df, nlp, stop_fr))
