# %% Dependicies
import polars as pl
import altair as alt
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

alt.data_transformers.enable("vegafusion")

# %% Read into dataframe

df = pl.read_csv("./../../data/v1/data.csv", ignore_errors=True)

# %% PCA

relevant_data = df["score", "descendants", "user_karma", "user_post_count"]
relevant_data = relevant_data.fill_null(0).fill_nan(0)

print(relevant_data.shape[0])


## Undersample (Does this make sense?)

low_scores = relevant_data.filter(pl.col("score") < 10).sample(fraction=0.2)
high_scores = relevant_data.filter(pl.col("score") >= 10)
relevant_data = pl.concat([low_scores, high_scores])

print(relevant_data.shape[0])


# %% PCA - Review
pca = PCA(n_components=2)
pca.fit(relevant_data.to_numpy())

print(pca.explained_variance_ratio_)
print(pca.components_)

## Result: Very little linearity between most features, PCA might be misleading

# %% Visualize relation between features

charts = []

for i in range(0, len(relevant_data.columns)):
    for j in range(i + 1, len(relevant_data.columns)):
        x = relevant_data.columns[i]
        y = relevant_data.columns[j]

        chart = (
            alt.Chart(relevant_data)
            .mark_circle()
            .encode(x=x, y=y, tooltip=[x, y])
            .properties(title=f"{x} vs {y}")
        )
        charts.append(chart)

charts[0] & charts[1] & charts[2] & charts[3] & charts[4]

# %% Spearmans correlation

for i in range(0, len(relevant_data.columns)):
    for j in range(i + 1, len(relevant_data.columns)):
        x = relevant_data.select(pl.nth(i)).to_numpy()
        y = relevant_data.select(pl.nth(j)).to_numpy()
        x_name = relevant_data.columns[i]
        y_name = relevant_data.columns[j]

        correlation, p_value = spearmanr(x, y)

        print(
            f"Spearman's rank correlation coefficient between {x_name} and {y_name} : {correlation}"
        )
        print(f"p-value: {p_value}")
