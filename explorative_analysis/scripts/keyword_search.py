# %% Load dependencies
import polars as pl
import altair as alt
import json


def analyse_feature(df, feature_list, feature):
    feature_dict = {}
    for feat in feature_list:
        feat = feat.lower()
        try:
            if feat:  # Ensure feat is not empty
                feature_dict[feat[0]].append(feat)
        except KeyError:
            feature_dict[feat[0]] = [feat]

    feature_detail = []
    contains_feature = []

    for row in df.iter_rows(named=True):
        if row["title"] is None:
            contains_feature.append(None)
            feature_detail.append(None)

            continue
        found = False
        for word in row["title"].split():
            if found:
                break
            word = word.lower()
            if word[0] in feature_dict:
                for feat in feature_dict[word[0]]:
                    if word == feat:

                        feature_detail.append(feat)
                        contains_feature.append(True)
                        found = True
                        break
        if not found:

            feature_detail.append(None)
            contains_feature.append(False)

    df = df.with_columns(pl.Series(f"{feature}_detail", feature_detail))
    df = df.with_columns(pl.Series(f"contains_{feature}", contains_feature))

    fig = (
        alt.Chart(
            df.filter(pl.col(f"contains_{feature}").is_not_null()),
            title="Contains " + feature,
        )
        .mark_boxplot(extent="min-max")
        .encode(
            y=f"contains_{feature}:N",
            x=alt.X("score:Q", scale=alt.Scale(type="log"), title="Score (log scale)"),
        )
    )
    fig.display()

    return df


# %% Load dataframe
alt.data_transformers.enable("vegafusion")

df = pl.read_csv("./../../data/v6/240min_data.csv", ignore_errors=True)


# %% Brand search with more detail
brands = pl.read_csv("./../../data/others/brands.csv", ignore_errors=True)
brands = brands["Name"].str.to_lowercase().to_list()

df = analyse_feature(df, brands, "brands")

# %% YC companies : Not working - not enough hits for yc companies

with open("./../../data/others/_yc_companies.json") as f:
    companies = json.load(f)

df = analyse_feature(df, companies, "yc_companies")

# %% Open source repositories
## TODO: Clean up repo list
with open("./../../data/others/repos.json") as f:
    repos = json.load(f)

df = analyse_feature(df, repos, "repos")

# %%
