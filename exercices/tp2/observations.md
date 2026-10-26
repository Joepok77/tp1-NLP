# TP2 : observations

**NLTK vs spaCy.** Les deux outils découpent le texte presque de la même façon. La différence vient des mots comme « l'icône » : NLTK les garde en un seul morceau, alors que spaCy sépare « l' » et « icône ». spaCy est donc mieux adapté au français.

**Lemmatisation.** Le pipeline garde l'essentiel : le premier ticket passe de 20 mots à 10, et le vocabulaire de chaque catégorie est divisé par deux. Le modèle fait quand même des erreurs : « passerelle » devient « passerell », « reste » n'est pas ramené à « rester », et « mqtt » est pris pour un verbe. On perd aussi la négation (« ne… plus ») et les chiffres (« version 3.2 »).

**Entités nommées.** Les villes sont bien reconnues (Lyon, Nantes, Toulouse). En revanche, les termes techniques sont mal classés : « MQTT » est pris pour une entreprise, « LoRaWAN » pour un lieu et « Cloud » pour une personne. Le modèle a appris sur du français courant et ne connaît pas le vocabulaire IoT.
