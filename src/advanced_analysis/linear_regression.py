# %%
import pandas as pd
import altair as alt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

score_name = "score_right"
lasso_do_log = True

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

y = selection[score_name].to_numpy()
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
"""Lasso regression"""
if lasso_do_log:
    y_lasso = np.log(y)
else:
    y_lasso = y
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_scaled, y_lasso)

print(f"Optimal alpha: {lasso_cv.alpha_}")

# Perform lasso regression
reg = linear_model.Lasso(alpha=lasso_cv.alpha_)

# R^2 score
# Will calculate score including training and test set
score = cross_val_score(reg, X_scaled, y_lasso, cv=3)
print("Cross-validated R^2 score:", score.mean())

# MSE
# Will calculate mse including training and test set
mse = cross_val_score(reg, X_scaled, y_lasso, cv=3, scoring="neg_mean_squared_error")
print("Cross-validated MSE:", -mse.mean())

# Check coefficients & indices of selected features
coefs = lasso_cv.coef_
selected_features = [i for i, coef in enumerate(coefs) if coef != 0]


# %%
"""Analyse linear model assumptions"""

reg.fit(X_scaled, y_lasso)

# 1. Mean residual
residuals = y_lasso - reg.predict(X_scaled)
mean_residual = np.mean(residuals)
print("Mean of residuals:", mean_residual)

# 2. Residual normality
# QQ plot
fig = sm.qqplot(residuals, line="s")
plt.show()

# Kolmogorov-Smirnov test
standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
kolmogorov_test = sm.stats.diagnostic.kstest_normal(standardized_residuals)
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
"""Poisson regression"""

reg = linear_model.PoissonRegressor()

# R^2 score
score = cross_val_score(reg, X_scaled, y, cv=3)
print("Cross-validated R^2 score:", score.mean())

# MSE
mse = cross_val_score(reg, X_scaled, y, cv=3, scoring="neg_mean_squared_error")
print("Cross-validated MSE:", -mse.mean())

"""Model assumptions of poisson regression"""
# 1. Data is count data --> is given: score is a count

# 2. Events are independent
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25)
reg.fit(X_train, y_train)
residuals = y_test - reg.predict(X_test)
dw_test = sm.stats.durbin_watson(residuals)
print("Durbin-Watson test:", dw_test)
# --> Nearly 2, so independence is given

# 3. Mean and variance are equal
mean_y = np.mean(y_train)
var_y = np.var(y_train)
print("Mean of y:", mean_y)
print("Variance of y:", var_y)
# --> Mean and variance are not equal, poisson regression is not appropriate
# Outliers effect poisson heavily, as no log transformation is applied
