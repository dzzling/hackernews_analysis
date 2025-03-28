# %% Dependicies
import polars as pl
import altair as alt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import shap
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_text
import numpy as np


alt.data_transformers.enable("vegafusion")

# TODO: Test model performance for classification

# %% Read into dataframe

df = pl.read_csv("./../../data/regression/data.csv", ignore_errors=True)

selection = (
    df[
        "score",
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

""" ## Undersample low scores
high_score = selection.filter(pl.col("score") > 2)
low_score = selection.filter(pl.col("score") < 2)
low_score_sampled = low_score.sample(n=len(high_score))

selection = pl.concat([low_score_sampled, high_score])
 """

# TODO: Perturb data set
y = selection["score"].to_numpy()
print(y.shape)
selection = selection.drop("score")
X = selection.to_numpy()
print(X.shape)

# y = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()
# X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# %% Simple Linear regression

clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)
coef = clf.coef_
print(coef)
intercept = clf.intercept_
print(intercept)

## Result: Still not good due to weak linearity between features

# %% Simple decision tree regression
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
score = regressor.score(X_test, y_test)
print(score)

tree_rules = export_text(regressor, feature_names=selection.columns)
print(tree_rules)

# %% Random forest regression

rf = RandomForestRegressor(
    n_estimators=800,  # More trees for stability
    max_depth=None,  # Limits complexity to avoid overfitting
    min_samples_split=5,  # Prevents small noisy splits
    min_samples_leaf=8,  # Ensures each leaf has enough data
    max_features="sqrt",  # more randomness or all?
    random_state=42,
    bootstrap=True,
)

cross_val_score(rf, X_train, y_train, cv=5)

# %% Difference between mean and median of tree predictions
rf.fit(X_train, y_train)

dict = {x: [] for x in range(10)}
for tree in range(800):
    pred = rf.estimators_[tree].predict(X_test[0:10])
    for i in range(10):
        dict[i].append(pred[i])

res_mead = []
res_mean = []
for x, y in dict.items():
    y.sort()
    res_mead.append(int(y[400]))
    res_mean.append(int(sum(y) / len(y)))

print("Mean and median of tree predictions")
print(res_mean)
print(res_mead)

error_mead = []
error_mean = []
for i in range(10):
    error_mead.append(abs(y_test[i] - res_mead[i]))
    error_mean.append(abs(y_test[i] - res_mean[i]))

print("Total absolute errors")
print(error_mean)
print(error_mead)

print("Mean error")
print(sum(error_mean) / len(error_mean))
print(sum(error_mead) / len(error_mead))

# %% Get RF feature importance

importances = rf.feature_importances_
print("Model feature importances:")
print(selection.columns)
print(importances)

## Permutation importance

print("Permutation importance:")
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_importances = result.importances_mean
perm_std = result.importances_std

print("Feature importances based on accuracy losses during feature permutation:")
print(perm_importances)
print(perm_std)

# %% SHAP
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# %%
