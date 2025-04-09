# %% Load dependencies
import polars as pl
import altair as alt
import json
from collections import Counter
import re


# %%
def analyse_title(df, feature_list, feature, counted=False):

    alt.data_transformers.enable("vegafusion")

    feature_dict = {}
    for feat in feature_list:
        feat = feat.lower()
        try:
            if isinstance(feat, str) and feat:  # Ensure feat is a non-empty string
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
        count = 0
        words_in_title = []

        for word in row["title"].split():
            word = word.lower()
            if word[0] in feature_dict:
                for feat in feature_dict[word[0]]:
                    if word == feat:
                        if counted:
                            count += 1
                            words_in_title.append(feat)
                        else:
                            feature_detail.append(feat)
                            contains_feature.append(1)
                            found = True
                            break
            if not counted and found:
                break

        if counted:
            feature_detail.append(str(words_in_title))
            contains_feature.append(count if count > 0 else 0)
        elif not found:
            feature_detail.append(None)
            contains_feature.append(0)

    print(Counter(contains_feature))

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
            tooltip=alt.Tooltip("mean(score)", title="Mean Score"),
        )
    )
    fig.display()

    return df


# %%
def analyse_url(df, feature_list, feature_name):
    x = "|".join(re.escape(feature.lower()) for feature in feature_list)

    if feature_name == "tech":
        print(x)

    feature_detail = []
    contains_feature = []

    for row in df.iter_rows(named=True):

        if row["url"] is None:
            feature_detail.append(None)
            contains_feature.append(0)
            continue

        match = re.match(
            f"^(?:https?:\/\/)?(?:www\.)?([a-zA-Z0-9-]+\.)*({x})\.", row["url"]
        )

        if (
            match
            and match.group(1) != "apps."
            and match.group(1) != "play."
            and match.group(1) != "chromewebstore."
        ):
            feature_detail.append(match.group(2))
            contains_feature.append(1)
        else:
            feature_detail.append(None)
            contains_feature.append(0)

    df = df.with_columns(
        pl.Series(f"{feature_name}_detail", feature_detail),
        pl.Series(f"contains_{feature_name}", contains_feature),
    )

    fig = (
        alt.Chart(
            df.filter(pl.col(f"contains_{feature_name}").is_not_null()),
            title="Contains " + feature_name,
        )
        .mark_boxplot(extent="min-max")
        .encode(
            y=f"contains_{feature_name}:N",
            x=alt.X("score:Q", scale=alt.Scale(type="log"), title="Score (log scale)"),
            tooltip=alt.Tooltip("mean(score)", title="Mean Score"),
        )
    )
    fig.display()

    return df


# %%
def infer_keywords(df):
    # %% Brand search with more detail
    brands = pl.read_csv("./../../data/others/brands.csv", ignore_errors=True)
    brands = brands["Name"].str.to_lowercase().to_list()

    df = analyse_title(df, brands, "brands")

    # %% YC companies : Not working - not enough hits for yc companies

    with open("./../../data/others/_yc_companies.json") as f:
        companies = json.load(f)

    df = analyse_title(df, companies, "yc_companies")

    # %% Open source repositories : Not working - not enough hits and too many repos named like regular english words
    with open("./../../data/others/repos.json") as f:
        repos = json.load(f)

    df = analyse_title(df, repos, "repos")

    # %% Politicians
    with open("./../../data/others/politicians.json") as f:
        politicians = json.load(f)

    df = analyse_title(df, politicians, "politicians")

    # %% Classical news domain
    with open("./../../data/others/classical_news.json") as f:
        classical_news = json.load(f)

    df = analyse_url(df, classical_news, "classical_news")

    # %% Startup news
    with open("./../../data/others/startup_news.json") as f:
        startup_news = json.load(f)

    df = analyse_url(df, startup_news, "startup_news")

    # %% Blog
    with open("./../../data/others/blog.json") as f:
        blogs = json.load(f)

    df = analyse_url(df, blogs, "blogs")

    # %% Academic
    with open("./../../data/others/academic.json") as f:
        academic = json.load(f)

    df = analyse_url(df, academic, "academic")

    # %% URL contains tech companies
    with open("./../../data/others/techcompanies.json") as f:
        tech = json.load(f)
        tech = [techcompany["name"] for techcompany in tech]

    df = analyse_url(df, tech, "tech")

    return df
