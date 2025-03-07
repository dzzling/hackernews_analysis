# %% Dependencies

import sqlite3
from transformers import pipeline
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import re
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import nltk

nltk.download("punkt_tab")
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
docs = [row[1] for row in rows]
""" 
docs = [doc for doc in docs if doc != ""]
docs = [doc.replace("\n", " ") for doc in docs]
docs = [doc.lower() for doc in docs]
docs = [" ".join(doc.split()) for doc in docs]
docs = [re.sub(r"[^a-z0-9 ]", "", doc) for doc in docs]

# Remove stopwords
stop_words = set(stopwords.words("english"))
docs = [
    " ".join([word for word in doc.split() if word not in stop_words]) for doc in docs
] """


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
# %% BERTopic

topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    representation_model=KeyBERTInspired(),
    min_topic_size=10,
    top_n_words=5,
)
topics, probs = topic_model.fit_transform(docs)

topic_model.get_topic_info()
# %%
# BERTopic Zero Shot

candidate_labels = [
    "Technology and Software Development",
    "Cybersecurity",
    "Artificial Intelligence and Machine Learning",
    "Startups and Entrepreneurship",
    "Science and Research",
    "Economics and Finance",
    "Politics and Society",
]

topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    min_topic_size=10,
    zeroshot_topic_list=candidate_labels,
    zeroshot_min_similarity=0.4,
    representation_model=KeyBERTInspired(),
    top_n_words=5,
)
topics, _ = topic_model.fit_transform(docs)
# %%
topic_model.get_topic_info()

# %%
topic_model.get_document_info(docs)

# %%

# LDA

docs = [word_tokenize(doc) for doc in docs]

dictionary = corpora.Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]

# %%
num_topics = 15  # Number of topics to extract
lda_model = models.LdaModel(
    corpus, num_topics=num_topics, id2word=dictionary, passes=15
)

print("Topics:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

# %%
