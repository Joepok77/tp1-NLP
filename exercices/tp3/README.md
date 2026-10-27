# Ressources — TP Séance 3 : Classification par prompt engineering

## Contenu du dossier

- `tickets_support_iot_v2.csv` — jeu de données (même corpus que la séance 2, 80 tickets).
- `tp_starter.py` — squelette de code avec les fonctions à compléter.
- `requirements.txt` — dépendances Python nécessaires.

## Installation (à faire avant la séance)

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration des clés d'API (à faire AVANT la séance)

Créez un fichier `.env` à la racine du dossier (jamais commité dans Git) :

```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

Puis chargez-le en tout début de script :

```python
from dotenv import load_dotenv
load_dotenv()
```

## Vérification rapide de l'environnement

```bash
python -c "import openai, google.genai; print('OK')"
```

⚠️ Ces API sont payantes au-delà d'un usage gratuit limité. Un échantillon de
15-20 tickets suffit largement pour réaliser tout le TP sans dépasser un budget
de quelques centimes.
