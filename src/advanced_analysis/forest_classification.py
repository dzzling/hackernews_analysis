# %%
# Load dependencies
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import polars as pl
import altair as alt
from tools import (
    parameter_tuning,
    get_decision_rules_from_forest,
)

alt.data_transformers.enable("vegafusion")
score_name = "score_right"

# %%
# Load and prepare data
df = pl.read_csv("./../../data/regression/data.csv", ignore_errors=True)

selection = (
    df[
        score_name,
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

y = (selection[score_name]).cut([2, 4], labels=["low", "medium", "high"]).to_numpy()
print(y)
X = selection.drop(score_name)
feature_names = X.columns
X = X.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# %%
# Random forest classifier
rf = RandomForestClassifier(
    **parameter_tuning(RandomForestClassifier(), y_train, X_train)
)

# %%
# Accuracy
scores = cross_val_score(rf, X_train, y_train, cv=3)
print("Cross-validation accuracy:", scores.mean())

# %%
# Get decision rules of the most successfull tree in the forest
rf.fit(X_train, y_train)
get_decision_rules_from_forest(
    rf,
    X_test,
    y_test,
    feature_names,
)

# %%
# Get RF feature importance
# measures which features are most important to reduce gini impurity (diversity in nodes)
# impurity-based feature importance can be biased towards features with high cardinality
importances = rf.feature_importances_
print("Model feature importances:")
importances = pl.DataFrame({"feature": feature_names, "importance": importances})
print(importances.sort("importance", descending=True))

# %%
# Permutation importance
# does not measure negative/positive influence on dependent variable - only measures impact on model performance
print("Permutation importance:")
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_importances = result.importances_mean
importances = pl.DataFrame({"feature": feature_names, "importance": perm_importances})
print(importances.sort("importance", descending=True))

# %%
# SHAP
# provides a measure of the impact of each feature on the model's predictions
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")

# %%
