# %% Dependencies

import sqlite3
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import re
from nltk.corpus import stopwords, words
import polars as pl
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
import sqlalchemy

# %%
# Database setup
DB_NAME = "./../data/scraped_data.db"
SOURCE_TABLE = "plain_webpages"

# Connect to SQLite
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute(f"SELECT id, plain FROM {SOURCE_TABLE}")
rows = cursor.fetchall()

# %% Text preprocessing
docs = [row[1] for row in rows]
ids = [row[0] for row in rows]


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

clean_pairs = [(id, doc.replace("\n", " ")) for (id, doc) in zip(ids, docs)]
clean_pairs = [(id, clean_text(doc)) for (id, doc) in clean_pairs]
clean_pairs = [(id, doc) for (id, doc) in clean_pairs if doc != ""]
docs = [doc for (id, doc) in clean_pairs]
ids = [id for (id, doc) in clean_pairs]

DEST_TABLE = "reduced_webpages"
reduced_df = pl.DataFrame({"Id": ids, "Document": docs})
reduced_df.write_database(
    table_name=DEST_TABLE,
    connection="sqlite:///./../data/scraped_data.db",
    if_table_exists="replace",
    engine="sqlalchemy",
)

# %% BERTopic

hdbscan_model = HDBSCAN(
    cluster_selection_epsilon=0.3,
    min_cluster_size=10,
    metric="euclidean",
    prediction_data=True,
)

kmeans_model = KMeans(n_clusters=10)

topic_model = BERTopic(
    representation_model=KeyBERTInspired(),
    hdbscan_model=kmeans_model,
)
topics, probs = topic_model.fit_transform(docs)
# %%
# Create a dataframe with document-topic pairs

topic_df = pl.from_pandas(topic_model.get_topic_info())
doc_topics_df = pl.DataFrame({"Topic": topics, "Document": docs, "Id": ids})
doc_topics_df = doc_topics_df.join(other=topic_df, on="Topic", how="inner")
doc_topics_df = doc_topics_df.drop("Count", "Representation", "Representative_Docs")
# %%
DEST_TABLE = "topic_webpages"
doc_topics_df.write_database(
    table_name=DEST_TABLE,
    connection="sqlite:///./../data/scraped_data.db",
    if_table_exists="replace",
    engine="sqlalchemy",
)

# %%
