"""
TP Seance 3 - Etape 5 : comparatif des approches de classification.

- calcule l'accuracy de TF-IDF + regression logistique sur les memes 15 tickets de test
- rassemble les resultats dans un tableau exporte en comparatif_approches.csv
"""

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DOSSIER = Path(__file__).parent

# Memes 15 tickets de test que dans tp_starter.py ; les 65 autres servent a l'entrainement
df = pd.read_csv(DOSSIER / "tickets_support_iot_v2.csv")
test = df.sample(n=15, random_state=42)
entrainement = df.drop(test.index)

# TF-IDF : chaque ticket devient un vecteur de poids de mots (comme l'exercice 3 du TP1)
vectorizer = TfidfVectorizer()
X_entrainement = vectorizer.fit_transform(entrainement["texte"])
X_test = vectorizer.transform(test["texte"])

# Un classifieur simple apprend a relier ces vecteurs aux categories
classifieur = LogisticRegression(max_iter=1000)
classifieur.fit(X_entrainement, entrainement["categorie"])
predictions = classifieur.predict(X_test)

acc_tfidf = (predictions == test["categorie"]).mean()
print("Accuracy TF-IDF :", round(acc_tfidf, 3))

# Tableau comparatif (accuracies LLM mesurees avec tp_starter.py sur les memes 15 tickets)
tableau = pd.DataFrame([
    {"approche": "TF-IDF + regression logistique", "accuracy": round(acc_tfidf, 3),
     "entrainement": "oui, quelques secondes sur CPU", "cout": "quasi nul (local)",
     "latence": "tres faible (millisecondes)", "gpu": "non"},
    {"approche": "BERT feature extraction", "accuracy": "non mesure (seance 2 non realisee)",
     "entrainement": "oui, classifieur leger sur les vecteurs BERT", "cout": "faible (local)",
     "latence": "faible a moyenne", "gpu": "conseille, pas obligatoire"},
    {"approche": "Zero-shot (Gemini)", "accuracy": 1.0,
     "entrainement": "aucun", "cout": "appel API payant au-dela du gratuit (~80 tokens/ticket)",
     "latence": "elevee (~1 a 2 s par ticket + limites de requetes)", "gpu": "non (cote fournisseur)"},
    {"approche": "Few-shot (Gemini)", "accuracy": 1.0,
     "entrainement": "aucun", "cout": "appel API, ~3,6x plus de tokens (~290/ticket)",
     "latence": "elevee", "gpu": "non (cote fournisseur)"},
    {"approche": "Chain-of-thought (Gemini)", "accuracy": 0.933,
     "entrainement": "aucun", "cout": "appel API, ~1,5x plus de tokens (~125/ticket)",
     "latence": "la plus elevee (reponses plus longues)", "gpu": "non (cote fournisseur)"},
])
print(tableau.to_string(index=False))

tableau.to_csv(DOSSIER / "comparatif_approches.csv", index=False)
print("\nTableau exporte : comparatif_approches.csv")
