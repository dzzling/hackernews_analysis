# %% Dependencies

import sqlite3
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import nltk
import gensim
import string
import numpy as np
from sklearn.cluster import KMeans
from nltk.corpus import stopwords, words
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# %%
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("words")

# %%
# Database setup
DB_NAME = "./../data/scraped_data.db"
SOURCE_TABLE = "plain_webpages"

# Connect to SQLite
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute(f"SELECT id, plain FROM {SOURCE_TABLE}")
rows = cursor.fetchall()
conn.close()

# %% Text preprocessing


def clean_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenize and filter non-English words
    tokens = text.lower().split()
    tokens = [
        word for word in tokens if word in english_words and word not in stop_words
    ]
    return " ".join(tokens)


english_words = set(words.words())
stop_words = set(stopwords.words("english"))

docs = [row[1] for row in rows]

docs = [doc for doc in docs if doc != ""]
docs = [doc.replace("\n", " ") for doc in docs]
docs = [clean_text(doc) for doc in docs]
docs = [word_tokenize(doc) for doc in docs]
# docs = [pos_tag(doc) for doc in docs]
# docs = [[word for (word, pos) in doc if pos == "NN" or pos == "NNP"] for doc in docs]

# %%
print(docs[0])
# %%
# %%
# Word2Vec

word2vec_model = Word2Vec(
    sentences=docs, vector_size=100, window=5, min_count=1, workers=4
)

# %%
from sklearn.cluster import KMeans

# Get unique words and their vectors
word_vectors = word2vec_model.wv
vocab = list(word_vectors.index_to_key)
X = np.array(
    [word_vectors[word] for word in vocab]
)  # Convert word vectors to NumPy array

# Cluster words using K-means
num_clusters = 10  # Adjust the number of topics
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
# %%
# Group words by clusters

# %%
print(len(vocab))
print(len(labels))

# %%
word_clusters = {}
for word, label in zip(vocab, labels):
    print(word, label)
    try:
        word_clusters[str(label)].append(word)
    except Exception:
        word_clusters[str(label)] = [word]

# %%
for key in word_clusters.keys():
    print(key)
# %%
# Display topics
for cluster in word_clusters.keys():
    print(f"Topic {cluster}: {word_clusters[cluster][:10]}")

# %%
