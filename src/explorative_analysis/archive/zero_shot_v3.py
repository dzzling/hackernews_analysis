# %% Dependencies

import sqlite3
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

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

docs = [doc for doc in docs if doc != ""]
docs = [doc.replace("\n", " ") for doc in docs]
docs = [doc.lower() for doc in docs]
docs = [" ".join(doc.split()) for doc in docs]
docs = [re.sub(r"[^a-z0-9 ]", "", doc) for doc in docs]

# Remove stopwords
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, min_df=2, max_features=1000, stop_words="english"
)
tfidf = tfidf_vectorizer.fit_transform(docs)
# %%
# Fit the NMF model
nmf = NMF(
    n_components=5,
    random_state=1,
    init="nndsvda",
    beta_loss="frobenius",
    alpha_W=0.00005,
    alpha_H=0.00005,
    l1_ratio=1,
)
nmf.fit(tfidf)

# Get the feature names
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()


# %%
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
    fig.suptitle(title, fontsize=40)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


# Plot the top words for each topic
plot_top_words(nmf, tfidf_feature_names, 20, "Topics in NMF model (Frobenius norm)")
