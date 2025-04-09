# %% Dependicies
import polars as pl
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import shap
from sklearn.model_selection import cross_val_score
import numpy as np
from tools import (
    simple_linear_regression,
    vectorize_and_clean_strings,
    scale_data,
    simple_decision_tree,
    parameter_tuning,
    get_decision_rules_from_forest,
    mean_vs_median,
    undersample,
    in_out_sample,
)

alt.data_transformers.enable("vegafusion")

# %% Read into dataframe

df = pl.read_csv(
    "./../../data/regression/data.csv",
    ignore_errors=True,
)

# %%

selection = (
    df[
        "title",
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

## Get vectorized titles
""" vectorized_words = vectorize_and_clean_strings(
    selection["title"].to_list(), vector_size=1
) """
selection = selection.drop("title")

## Undersample low scores
selection = undersample(selection, limit=2)

## Get target
y = selection["score"].to_numpy()

## Remove target from features
selection = selection.drop("score")

## Include vectorized titles as features
X = selection.to_numpy()
# X = np.hstack((X, vectorized_words))

""" feature_names = selection.columns + [
    "title_vector_" + str(i) for i in range(vectorized_words.shape[1])
] """
feature_names = selection.columns
print(X.shape)

## Scale data
# X, y = scale_data(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# %% Simple Linear regression
simple_linear_regression(X_train, y_train, X_test, y_test)
## Result: Still not good (due to weak linearity between features?)

# %% Simple decision tree regression
simple_decision_tree(X_train, y_train, X_test, y_test)

# %% Random forest regression

rf = RandomForestRegressor(
    n_estimators=100,  # more trees for stability
    max_depth=10,  # limits complexity to avoid overfitting
    min_samples_split=10,  # prevents small noisy splits
    min_samples_leaf=5,  # ensures each leaf has enough data
    max_features="sqrt",  # more randomness or all?
)
print(cross_val_score(rf, X, y, cv=3))

# %% Hyperparameter tuning
parameter_tuning(rf, y_train, X_train)

# %% Get decision rules of the most successfull tree in the forest
get_decision_rules_from_forest(
    rf,
    X_test,
    y_test,
    feature_names,
)

# %% Difference between mean and median of tree predictions
rf.fit(X_train, y_train)
mean_vs_median(rf, X_train, y_train, X_test, y_test)

# %% Get RF feature importance
importances = rf.feature_importances_
print("Model feature importances:")
importances = pl.DataFrame({"feature": feature_names, "importance": importances})
print(importances.sort("importance", descending=True))

# %% Permutation importance
print("Permutation importance:")
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_importances = result.importances_mean
importances = pl.DataFrame({"feature": feature_names, "importance": perm_importances})
print(importances.sort("importance", descending=True))

# %% SHAP
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")

# %% Overfitting Random Forest
rf_overfit = RandomForestRegressor(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=selection.shape[1] - 5,
    random_state=42,
)
# %% In- vs. out-Sample Prediction
print("Without overfitting:")
in_out_sample(rf, X_train, y_train, X_test, y_test)

print("With overfitting:")
in_out_sample(rf_overfit, X_train, y_train, X_test, y_test)
# %%
