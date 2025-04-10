from transformers import pipeline
import polars as pl
import altair as alt

df = pl.read_csv("./../../data/v6/30min_data.csv").filter(pl.col("title").is_not_null())
titles = df["title"].to_list()


classifier = pipeline(
    "zero-shot-classification", model="knowledgator/comprehend_it-base"
)

candidate_labels = [
    "gratifies intellectual curiosity",
    "does not gratify intellectual curiosity",
]
results = [classifier(doc, candidate_labels) for doc in titles[:500]]

for result in results:
    print(result["sequence"], ":", result["labels"][0], result["scores"][0])

one_hot = [
    1 if result["labels"][0] == "gratifies intellectual curiosity" else 0
    for result in results
]

df_filtered = df.head(500).with_columns(
    pl.Series("gratifies_intellectual_curiosity", one_hot)
)

fig = (
    alt.Chart(
        df_filtered.filter(pl.col("gratifies_intellectual_curiosity").is_not_null()),
        title="gratifies_curiosity",
    )
    .mark_boxplot(extent="min-max")
    .encode(
        y="gratifies_intellectual_curiosity:N",
        x=alt.X("score:Q", scale=alt.Scale(type="log"), title="Score (log scale)"),
    )
)
fig

# %%
