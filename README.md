# TP01 - Natural Language Processing avec Python

## Installation

Récupérer le projet :

```bash
git clone https://github.com/Joepok77/tp1-NLP.git
cd tp1-NLP
```

Créer l'environnement virtuel (une seule fois) :

```bash
python -m venv .venv
```

L'activer (à refaire à chaque nouveau terminal) :

| Terminal | Commande |
| --- | --- |
| Git Bash (Windows) | `source .venv/Scripts/activate` |
| PowerShell (Windows) | `.venv\Scripts\Activate.ps1` |
| Linux / macOS | `source .venv/bin/activate` |

Une fois activé, `(.venv)` apparaît au début de la ligne du terminal.

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Pour quitter l'environnement virtuel :

```bash
deactivate
```

## Exercice 1 : prétraitement de texte

Télécharger les données NLTK (une seule fois) :

```bash
python -m nltk.downloader punkt_tab stopwords wordnet averaged_perceptron_tagger_eng
```

Lancer l'exercice :

```bash
python exercices/tp1/exercice1.py
```

Le script applique quatre étapes au texte du sujet avec NLTK :

| Étape | Outil NLTK | Ce que ça fait | Exemple |
| --- | --- | --- | --- |
| Tokenization | `word_tokenize` | sépare le texte en mots, la ponctuation compte comme un mot | 27 tokens |
| Stemming | `PorterStemmer` | coupe la fin des mots avec des règles, sans dictionnaire | `Processing` → `process`, `languages` → `languag` |
| Lemmatization | `WordNetLemmatizer` + `pos_tag` | donne la forme du dictionnaire | `is` → `be`, `machines` → `machine` |
| Stop words | `stopwords` | retire les mots très fréquents qui portent peu de sens | `is`, `an`, `It`, `to`, `and`, `more` |

Le stemming peut donner des racines qui ne sont pas de vrais mots (`languag`, `variou`), alors que la lemmatization renvoie toujours un mot existant. En contrepartie, la lemmatization a besoin de connaître le type de chaque mot (verbe, nom, adjectif) : c'est le rôle de `pos_tag`, sans lequel `is` resterait `is`.
