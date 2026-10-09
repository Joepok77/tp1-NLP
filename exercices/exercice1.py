from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer 
import nltk
from nltk.corpus import stopwords


# Tokenization
text = "Natural Language Processing is an exciting field! It enables machines to understand human languages and perform various tasks like translation, summarization, and more."
tokens = word_tokenize(text)
print(tokens)

# Stemming

porter_stemmer = PorterStemmer()

stemmed_words = [porter_stemmer.stem(token) for token in tokens]

print("Original words:", tokens)
print("Stemmed words:", stemmed_words)

# Lemmatization

lemmatizer = WordNetLemmatizer()

def get_pos(tag):                                     
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"

tags = nltk.pos_tag(tokens) 

lemmatized_words = [lemmatizer.lemmatize(token, get_pos(tag)) for token, tag in tags]

print(f"Original Text: {text}")
print(f"Lemmatized Words: {lemmatized_words}")


# stop word

stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

print("Original:", tokens)
print("Filtered:", filtered_tokens)