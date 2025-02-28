# %% Dependicies
import polars as pl
import altair as alt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

alt.data_transformers.enable("vegafusion")

# %% Read into dataframe

df = pl.read_csv("./../../data/v1/data.csv", ignore_errors=True)

X = df["descendants", "user_karma", "user_post_count"]
X = X.fill_null(0).fill_nan(0).to_numpy()
print(X.shape)
y = df["score"].to_numpy()
print(y.shape)

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
rf_default = RandomForestRegressor(random_state=42)
rf_default.fit(X_train, y_train)

rf_default_score = rf_default.score(X_test, y_test)
print(rf_default_score)

# ?Cluster analysis, to detect if samples with similar feature combinations also have similar scores?

# %%
