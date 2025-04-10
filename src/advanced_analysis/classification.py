# %% Load dependencies
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import polars as pl
import altair as alt
from tools import (
    in_out_sample,
)

alt.data_transformers.enable("vegafusion")

# %% Load and prepare data
df = pl.read_csv("./../../data/regression/data.csv", ignore_errors=True)
fpdf = pl.read_csv("./../../data/v7/front_page_data.csv", ignore_errors=True)

selection = (
    df[
        "id",
        "score",
        "user_karma",
        "user_post_count",
        "is_monday",
        "is_tuesday",
        "is_wednesday",
        "is_thursday",
        "is_friday",
        "is_saturday",
        "is_sunday",
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
        "contains_classical_news",
        "contains_startup_news",
        "contains_blogs",
        "contains_academic",
        "contains_tech",
        "topic_0",
        "topic_1",
        "topic_2",
        "topic_3",
        "topic_4",
        "topic_5",
        "topic_6",
        "topic_7",
        "topic_8",
        "topic_9",
        "document_length",
    ]
    .drop_nulls()
    .drop_nans()
)

selection = selection.with_columns(
    pl.when(selection["id"].is_in(fpdf["id"]))
    .then(pl.lit("yes"))
    .otherwise(pl.lit("no"))
    .alias("on_frontpage")
)

## Undersample low scores
high_score = selection.filter(pl.col("score") > 4)
medium_score = selection.filter((pl.col("score") > 1) & (pl.col("score") <= 4))
low_score = selection.filter(pl.col("score") <= 1)
medium_score = medium_score.sample(n=len(high_score))
low_score_sampled = low_score.sample(n=len(high_score))
selection = pl.concat([low_score_sampled, medium_score, high_score])

# Save unsampled data for testing
unsampled = low_score.join(low_score_sampled, on="id", how="anti")

print("Data shape:")
y = (selection["score"]).cut([2, 4], labels=["low", "medium", "high"]).to_numpy()
print("Y: " + str(y.shape))
y_test_extended = (
    unsampled["score"].cut([2, 4], labels=["low", "medium", "high"]).to_numpy()
)

y_other = selection["on_frontpage"].to_numpy()
print("Y: " + str(y.shape))
y_test_extended_other = unsampled["on_frontpage"].to_numpy()

selection = selection.drop("score", "id", "on_frontpage")
X = selection.to_numpy()
print("X: " + str(X.shape))
X_test_extended = unsampled.drop("score", "id", "on_frontpage").to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# %%
clf = RandomForestClassifier(
    n_estimators=100,  # more trees for stability
    max_depth=10,  # limits complexity to avoid overfitting
    min_samples_split=10,  # prevents small noisy splits
    min_samples_leaf=5,  # ensures each leaf has enough data
    max_features="sqrt",  # more randomness or all?
    random_state=42,
)
cross_val_score(clf, X, y, cv=5)

# %%
indices = random.sample(range(len(y_test)), 10)
clf.fit(X_train, y_train)
print(y_test[indices])
res = clf.predict(X_test)
print(res[indices])
score = clf.score(X_test, y_test)
print(score)

# %% Overfitted random forest

clf_overfitted = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=selection.shape[1] - 5,
    random_state=42,
)

# %% In vs. out of sample
print("Regular random forest")
in_out_sample(clf, X_train, y_train, X_test, y_test)
print("Overfitted random forest")
in_out_sample(clf_overfitted, X_train, y_train, X_test, y_test)

# %% Made-it-to-front-page classification
cross_val_score(clf, X, y_other, cv=5)

# %% Examine some results
X_train, X_test, y_train_other, y_test_other = train_test_split(
    X, y_other, test_size=0.25
)
clf.fit(X_train, y_train_other)
y_pred = clf.predict(X_test)
print(y_test_other[:50])
print(y_pred[:50])

# %%
