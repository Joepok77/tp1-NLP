import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Représentation TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)          # matrice documents x mots
vocab = vectorizer.get_feature_names_out()
vecteurs_mots = X.T.toarray()                 # chaque mot devient un vecteur sur les documents
index = {mot: i for i, mot in enumerate(vocab)}

def tfidf_similaires(mot, topn=5):
    sims = cosine_similarity(vecteurs_mots[index[mot]].reshape(1, -1), vecteurs_mots)[0]
    ordre = np.argsort(sims)[::-1]
    return [(vocab[i], round(sims[i], 3)) for i in ordre if vocab[i] != mot][:topn]

def tfidf_similarite(m1, m2):
    v1 = vecteurs_mots[index[m1]].reshape(1, -1)
    v2 = vecteurs_mots[index[m2]].reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]

print("Mots les plus similaires à 'language'")
print("TF-IDF   :", tfidf_similaires("language"))
print("Word2Vec :", model.wv.most_similar("language", topn=5))

print("\nSimilarité entre 'machine' et 'learning'")
print("TF-IDF   :", round(tfidf_similarite("machine", "learning"), 3))
print("Word2Vec :", round(float(model.wv.similarity("machine", "learning")), 3))