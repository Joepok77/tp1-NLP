# Ressources — TP Séance 1 : Pipeline NLP complet avec spaCy et NLTK

## Contenu du dossier

- `tickets_support_iot.csv` — jeu de données : 50 tickets de support client d'une entreprise
  industrielle qui déploie des capteurs IoT (connectivité, batterie, firmware, sécurité,
  installation, interface). Colonnes : `id`, `categorie`, `texte`.
- `tp_starter.py` — squelette de code avec les fonctions à compléter, une par étape du TP.
- `requirements.txt` — dépendances Python nécessaires.

## Installation (à faire avant la séance)

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
python -m nltk.downloader stopwords punkt punkt_tab averaged_perceptron_tagger
```

## Vérification rapide de l'environnement

```bash
python -c "import spacy, nltk, pandas; nlp = spacy.load('fr_core_news_sm'); print('OK :', nlp('Bonjour le monde'))"
```

Si cette commande affiche une phrase tokenizée sans erreur, l'environnement est prêt pour le TP.
