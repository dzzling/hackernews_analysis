# %% Dependencies

import sqlite3
from transformers import pipeline
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import re
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import polars as pl
from hdbscan import HDBSCAN

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
english_words = {x for x in english_words if len(x) > 1}
stop_words = set(stopwords.words("english"))

docs = [doc for doc in docs if doc != ""]
docs = [doc.replace("\n", " ") for doc in docs]
docs = [clean_text(doc) for doc in docs]
docs = [doc for doc in docs if doc != ""]
# %% BERTopic

hdbscan_model = HDBSCAN(
    cluster_selection_epsilon=0.4, metric="euclidean", prediction_data=True
)

topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    representation_model=KeyBERTInspired(),
    top_n_words=10,
    hdbscan_model=hdbscan_model,
)
topics, probs = topic_model.fit_transform(docs)
topics_df = pl.DataFrame({"topic": topics, "document": docs})

topic_model.get_topic_info()

# %%

# LDA
tokenized_docs = [word_tokenize(doc) for doc in docs]
dictionary = corpora.Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

# %%
num_topics = 6  # Number of topics to extract
lda_model = models.LdaModel(
    corpus, num_topics=num_topics, id2word=dictionary, passes=15
)

print("Topics:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

# %%
