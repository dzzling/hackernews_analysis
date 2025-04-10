# %% Dependencies

import sqlite3
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import re
from nltk.corpus import stopwords, words
import polars as pl
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

# %%
# Database setup
DB_NAME = "./../../data/v7/scraped_data.db"
SOURCE_TABLE = "plain_webpages"

# Connect to SQLite
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute(f"SELECT id, plain FROM {SOURCE_TABLE}")
rows = cursor.fetchall()

# %% Text preprocessing
docs = [row[1] for row in rows]
ids = [row[0] for row in rows]

docs_ids_df = pl.DataFrame({"id": ids, "doc": docs})
df_with_titles = pl.read_csv("./../../data/v7/30min_data.csv")
data = docs_ids_df.join(df_with_titles, on="id", how="right")
data = data.filter(data["title"].is_not_null())

docs = data["doc"].to_list()
ids = data["id"].to_list()
titles = data["title"].to_list()
text = data["text"].to_list()


def clean_text(text):
    if text is None:
        return " "

    # Remove non-alphabetic characters
    text = text.replace("\n", " ")
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

# Clean text
clean_pairs = [
    (id, clean_text(doc), clean_text(text), title)
    for (id, doc, text, title) in zip(ids, docs, text, titles)
]
length = [
    3 if len(doc) > 200 else 2 if len(doc) > 50 else 1
    for (_, doc, text, title) in clean_pairs
]

# Combine doc and title
docs = [title + " " + text + " " + doc for (_, doc, text, title) in clean_pairs]
ids = [id for (id, _, _, _) in clean_pairs]

# %%

DEST_TABLE = "reduced_webpages"
reduced_df = pl.DataFrame({"Id": ids, "Document": docs, "Length": length})
reduced_df.write_database(
    table_name=DEST_TABLE,
    connection="sqlite:///./../../data/v7/scraped_data.db",
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
doc_topics_df = pl.DataFrame(
    {"Topic": topics, "Document": docs, "Id": ids, "Length": length}
)
doc_topics_df = doc_topics_df.join(other=topic_df, on="Topic", how="inner")
doc_topics_df = doc_topics_df.drop("Count", "Representation", "Representative_Docs")
# %%
DEST_TABLE = "topic_webpages"
doc_topics_df.write_database(
    table_name=DEST_TABLE,
    connection="sqlite:///./../../data/v7/scraped_data.db",
    if_table_exists="replace",
    engine="sqlalchemy",
)

# %%
