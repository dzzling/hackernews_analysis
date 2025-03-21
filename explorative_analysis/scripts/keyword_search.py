# %% Load dependencies
import polars as pl
import altair as alt
import json


def analyse_feature(df, feature_list, feature):
    feature_dict = {}
    for feat in feature_list:
        feat = feat.lower()
        try:
            feature_dict[feat[0]].append(feat)
        except KeyError:
            feature_dict[feat[0]] = [feat]

    feature_detail = []
    feature = []

    for row in df.iter_rows(named=True):
        if row["title"] is None:
            feature.append(None)
            feature_detail.append(None)
            continue
        found = False
        for word in row["title"].split():
            if found:
                break
            word = word.lower()
            if word[0] in feature_dict:
                for comp in feature_dict[word[0]]:
                    if word == comp:
                        feature_detail.append(comp)
                        feature.append(True)
                        found = True
    if not found:
        feature_detail.append(None)
        feature.append(False)

    df = df.with_columns(
        pl.Series(f"{feature}_detail", feature_detail),
        pl.Series(f"{feature}", feature),
    )

    fig = (
        alt.Chart(
            df.filter(pl.col(f"{feature}").is_not_null()),
            title="Contains " + feature,
        )
        .mark_boxplot(extent="min-max")
        .encode(
            y=f"{feature}:N",
            x=alt.X("score:Q", scale=alt.Scale(type="log"), title="Score (log scale)"),
        )
    )
    fig


# %% Load dataframe
alt.data_transformers.enable("vegafusion")

df = pl.read_csv("./../../data/v6/240min_data.csv", ignore_errors=True)


# %% Brand search with more detail
brands = pl.read_csv("./../../data/others/brands.csv", ignore_errors=True)
brands = brands["Name"].to_list()
brands = [brand.lower() for brand in brands]

brand_dict = {}
for brand in brands:
    brand = brand.lower()
    try:
        brand_dict[brand[0]].append(brand)
    except KeyError:
        brand_dict[brand[0]] = [brand]

contains_brand_detail = []
contains_brand = []


for row in df.iter_rows(named=True):
    if row["title"] is None:
        contains_brand.append(None)
        contains_brand_detail.append(None)
        continue
    found = False
    for word in row["title"].split():
        if found:
            break
        word = word.lower()
        if word[0] in brand_dict:
            for brand in brand_dict[word[0]]:
                if word == brand:
                    contains_brand_detail.append(brand)
                    contains_brand.append(True)
                    found = True
    if not found:
        contains_brand_detail.append(None)
        contains_brand.append(False)

df = df.with_columns(
    pl.Series("contains_brand_detail", contains_brand_detail),
    pl.Series("contains_brand", contains_brand),
)

fig = (
    alt.Chart(
        df.filter(pl.col("contains_brand").is_not_null()),
        title="Contains brand",
    )
    .mark_boxplot(extent="min-max")
    .encode(
        y="contains_brand:N",
        x=alt.X("score:Q", scale=alt.Scale(type="log"), title="Score (log scale)"),
    )
)
fig

# %% YC companies : Not working - not enough hits for yc companies

with open("./../../data/others/_yc_companies.json") as f:
    companies = json.load(f)
company_dict = {}
for comp in companies:
    comp = comp.lower()
    try:
        company_dict[comp[0]].append(comp)
    except KeyError:
        company_dict[comp[0]] = [comp]

contains_comp_detail = []
contains_comp = []


for row in df.iter_rows(named=True):
    if row["title"] is None:
        contains_comp.append(None)
        contains_comp_detail.append(None)
        continue
    found = False
    for word in row["title"].split():
        if found:
            break
        word = word.lower()
        if word[0] in company_dict:
            for comp in company_dict[word[0]]:
                if word == comp:
                    contains_comp_detail.append(comp)
                    contains_comp.append(True)
                    found = True
    if not found:
        contains_comp_detail.append(None)
        contains_comp.append(False)

df = df.with_columns(
    pl.Series("contains_comp_detail", contains_comp_detail),
    pl.Series("contains_comp", contains_comp),
)

fig = (
    alt.Chart(
        df.filter(pl.col("contains_comp").is_not_null()),
        title="Contains yc company",
    )
    .mark_boxplot(extent="min-max")
    .encode(
        y="contains_comp:N",
        x=alt.X("score:Q", scale=alt.Scale(type="log"), title="Score (log scale)"),
    )
)
fig

# %% Open source repositories
with open("./../../data/others/repos.json") as f:
    repos = json.load(f)
