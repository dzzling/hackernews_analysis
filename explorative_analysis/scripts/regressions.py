# %% Dependicies
import polars as pl
import altair as alt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import shap

alt.data_transformers.enable("vegafusion")

# TODO: Scaling?

# %% Read into dataframe

df = pl.read_csv("./../../data/v1/data.csv", ignore_errors=True)

features = df["descendants", "user_karma", "user_post_count"]
X = features.fill_null(0).fill_nan(0).to_numpy()
print(X.shape)
y = df["score"].to_numpy()
print(y.shape)

# %% Simple Linear regression

clf = linear_model.LinearRegression()
clf.fit(X, y)
score = clf.score(X, y)
print(score)
coef = clf.coef_
print(coef)
intercept = clf.intercept_
print(intercept)

## Result: Still not good due to weak linearity between features

# %% Random forest regression

rf = RandomForestRegressor(
    n_estimators=300,  # More trees for stability
    max_depth=10,  # Limits complexity to avoid overfitting
    min_samples_split=10,  # Prevents small noisy splits
    min_samples_leaf=5,  # Ensures each leaf has enough data
    max_features="sqrt",  # Balanced feature selection
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
rf.fit(X_train, y_train)

# %% RF Evaluation

score = rf.score(X_test, y_test)
print(score)

# %% RF Default
""" rf_default = RandomForestRegressor(random_state=42)
rf_default.fit(X_train, y_train)

rf_default_score = rf_default.score(X_test, y_test)
print(rf_default_score) """

# %% Get RF feature importance

importances = rf.feature_importances_
print("Model feature importances:")
print(features.columns)
print(importances)

## Permutation importance

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
