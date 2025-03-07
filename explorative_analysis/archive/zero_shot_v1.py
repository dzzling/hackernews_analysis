# %% Dependencies

import sqlite3
from transformers import pipeline
import re
from nltk.corpus import stopwords

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
docs = [row[1] for row in rows[:1000]]

docs = [doc for doc in docs if doc != ""]
docs = [doc.replace("\n", " ") for doc in docs]
docs = [doc.lower() for doc in docs]
docs = [" ".join(doc.split()) for doc in docs]
docs = [re.sub(r"[^a-z0-9 ]", "", doc) for doc in docs]

# Remove stopwords
stop_words = set(stopwords.words("english"))
docs = [
    " ".join([word for word in doc.split() if word not in stop_words]) for doc in docs
]

# %% Zero-shot classification using transformers pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

candidate_labels = [
    "Technology and Software Development",
    "Cybersecurity",
    "Artificial Intelligence and Machine Learning",
    "Startups and Entrepreneurship",
    "Science and Research",
    "Economics and Finance",
    "Politics and Society",
]
# %%

results = [classifier(doc, candidate_labels) for doc in docs[:30]]
# %%
for result in results:
    print(result["labels"][0], result["scores"][0], result["sequence"])

# %%
