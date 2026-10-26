import gensim.downloader as api
from gensim.models.word2vec import Text8Corpus
from gensim.models import Word2Vec

# Corpus Text8

chemin = api.load("text8", return_path=True)
corpus = Text8Corpus(chemin)

# Word2Vec

model = Word2Vec(corpus, vector_size=100, window=5, min_count=5, workers=4)

print(model)

# Most similar words

similar_words = model.wv.most_similar("language")

print("Most similar to language:", similar_words)

# Similarity

similarity = model.wv.similarity("machine", "learning")

print("Similarity machine / learning:", similarity)

# Analogy

analogy = model.wv.most_similar(positive=["king", "woman"], negative=["man"])

print("king - man + woman:", analogy)

