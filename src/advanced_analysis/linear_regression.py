# %%
import pandas as pd
import altair as alt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

score_name = "score_right"
do_log = True

# %%
df = pd.read_csv("./../../data/regression/data.csv")
alt.data_transformers.enable("vegafusion")

# Dropped one dummy variable for each category
selection = df[
    [
        "score_right",
        "user_karma",
        "user_post_count",
        "is_monday",
        "is_tuesday",
        "is_wednesday",
        "is_thursday",
        "is_friday",
        "is_saturday",
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
        "document_length",
    ]
].dropna()

y = selection[score_name]
if do_log:
    y = np.log(y)

X = selection.drop(score_name, axis=1)

# %%
# Correlation matrix
correlation_matrix = X.corr()

feat1, feat2, mag = [], [], []
columns = X.columns
for col1 in columns:
    for col2 in columns:
        feat1.append(col1)
        feat2.append(col2)
        mag.append(correlation_matrix.loc[col1, col2])

corr_df = pd.DataFrame(
    {
        "x": feat1,
        "y": feat2,
        "z": mag,
    }
)

corr_chart = (
    alt.Chart(corr_df)
    .mark_rect()
    .encode(
        x="x:N",
        y="y:N",
        color="z:Q",
    )
)
corr_chart

# %% Prepare data for regression
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# Fit Lasso with cross-validation to find best alpha
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_scaled, y)

print(f"Optimal alpha: {lasso_cv.alpha_}")

# Perform lasso regression
clf = linear_model.Lasso(alpha=lasso_cv.alpha_)

# R^2 score
score = cross_val_score(clf, X_scaled, y, cv=3)
print("Cross-validated R^2 score:", score.mean())

# MSE
mse = cross_val_score(clf, X_scaled, y, cv=3, scoring="neg_mean_squared_error")
print("Cross-validated MSE:", -mse.mean())

# Check coefficients & indices of selected features
coefs = lasso_cv.coef_
selected_features = [i for i, coef in enumerate(coefs) if coef != 0]


# %%
# Analyse linear model assumptions
clf.fit(X_scaled, y)

# 1. Mean residual
residuals = y - clf.predict(X_scaled)
mean_residual = np.mean(residuals)
print("Mean of residuals:", mean_residual)

# 2. Residual normality
# QQ plot
fig = sm.qqplot(residuals, line="s")
plt.show()

# Kolmogorov-Smirnov test
kolmogorov_test = sm.stats.diagnostic.kstest_normal(residuals)
print("Kolmogorov-Smirnov test:", kolmogorov_test)
# Result: p < 0.05: Null hypothesis rejected, residuals are not normally distributed

# 3. Homoscedasticity
# Breusch-Pagan test
temp = sm.add_constant(X_scaled)
bp_test = sm.stats.diagnostic.het_breuschpagan(residuals, temp)
print("Breusch-Pagan test:", bp_test)
# LM p-value <0.05, F p-value <0.05: Null hypothesis rejected, residuals are not homoscedastic

# 4. Independence
# Durbin-Watson test
dw_test = sm.stats.durbin_watson(residuals)
print("Durbin-Watson test:", dw_test)
# Result: around 2, almost perfect

# 5. Multicollinearity
# IVF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
# Some higher than 10, but not too many, lasso regression should help with that

# Log transformation aids with normality (0.37 -> 0.11 ks test) and homoscedasticity (79.14 -> 141.18 lagrange multiplier test)
# %%
