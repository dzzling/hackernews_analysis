# %% Load dependencies
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import polars as pl
import altair as alt
import numpy as np

alt.data_transformers.enable("vegafusion")

# %% Load and prepare data
df = pl.read_csv("./../../data/regression/data.csv", ignore_errors=True)

selection = (
    df[
        "id",
        "score_right",
        "user_karma",
        "user_post_count",
        "weekday",
        "hour",
        "title_length",
        "body_length",
        "time_on_first_page",
        "average_score_of_posts_before",
        "hourly_median_fp_score",
        "count_last_12h",
        "count_last_48h",
        "contains_brands",
        "contains_yc_companies",
        "contains_repos",
        "contains_politicians",
        "contains_buzzwords",
        "topic",
        "document_length",
    ]
    .drop_nulls()
    .drop_nans()
)

## Undersample low scores
high_score = selection.filter(pl.col("score_right") > 4)
low_score = selection.filter(pl.col("score_right") <= 4)
low_score_sampled = low_score.sample(n=len(high_score) // 2)
selection = pl.concat([low_score_sampled, high_score])

# Save unsampled data for testing
unsampled = low_score.join(low_score_sampled, on="id", how="anti")

print("Data shape:")
y = (selection["score_right"]).cut([2, 10], labels=["low", "medium", "high"]).to_numpy()
print("Y: " + str(y.shape))
y_test_extended = (
    unsampled["score_right"].cut([2, 10], labels=["low", "medium", "high"]).to_numpy()
)
selection = selection.drop("score_right")
X = selection.to_numpy()
print("X: " + str(X.shape))
X_test_extended = unsampled.drop("score_right").to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# %%
clf = RandomForestClassifier()
cross_val_score(clf, X, y, cv=5)

# %%
clf.fit(X_train, y_train)
res = clf.predict(np.concatenate((X_test, X_test_extended)))
print(res[len(res) - 10 :])
print(y_test_extended[len(y_test_extended) - 10 :])
score = clf.score(
    np.concatenate((X_test, X_test_extended)), np.concatenate((y_test, y_test_extended))
)
print(score)

# %%
