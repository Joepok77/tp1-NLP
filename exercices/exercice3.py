import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Même corpus que l'exercice 2 
documents = [" ".join(morceau) for morceau in corpus]
print("Nombre de documents :", len(documents))

# 2a. Représentation TF-IDF
vectorizer = TfidfVectorizer(min_df=5)
X = vectorizer.fit_transform(documents)       # documents x mots (matrice creuse)
vocab = vectorizer.get_feature_names_out()
vecteurs_mots = X.T.tocsr()                   # mots x documents, toujours creuse
index = {mot: i for i, mot in enumerate(vocab)}
print("Taille du vocabulaire TF-IDF :", len(vocab))

def tfidf_similaires(mot, topn=10):
    sims = cosine_similarity(vecteurs_mots[index[mot]], vecteurs_mots)[0]
    ordre = np.argsort(sims)[::-1]
    return [(vocab[i], round(float(sims[i]), 3)) for i in ordre if vocab[i] != mot][:topn]

def tfidf_similarite(m1, m2):
    return float(cosine_similarity(vecteurs_mots[index[m1]], vecteurs_mots[index[m2]])[0][0])

# 2b. Représentation Word2Vec : le modèle entraîné à l'exercice 2 (model)

# 3. Comparaison
print("\nMots les plus similaires à 'language'")
print("TF-IDF   :", tfidf_similaires("language"))
print("Word2Vec :", model.wv.most_similar("language", topn=10))

print("\nSimilarité entre 'machine' et 'learning'")
print("TF-IDF   :", round(tfidf_similarite("machine", "learning"), 3))
print("Word2Vec :", round(float(model.wv.similarity("machine", "learning")), 3))