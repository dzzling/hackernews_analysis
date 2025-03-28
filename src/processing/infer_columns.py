# %%
# Dependencies

import polars as pl
from keyword_search import analyse_title, infer_keywords
import sqlite3
from nltk.corpus import stopwords
from collections import Counter

# %%
# Read into dataframe

df1 = pl.read_csv("../../data/v7/30min_data.csv", ignore_errors=True)
df2 = pl.read_csv("../../data/v7/240min_data.csv", ignore_errors=True)
df3 = pl.read_csv("../../data/v7/front_page_data.csv", ignore_errors=True)
df4 = pl.read_csv("../../data/v7/second_chance_data.csv", ignore_errors=True)
# %%

df1 = df1.join(df2, on="id", how="inner")

# %%
# Infer time in hours scraped (unix)
df3 = df3.with_columns(
    ((pl.col("time") + (pl.col("minutes_since_pub") * 60)) // 3600).alias(
        "scraped_at_hour"
    )
)

# %%
# Infer weekday from time
df1 = df1.with_columns(((((pl.col("time") - 32400) // 86400) + 4) % 7).alias("weekday"))

# %%
# Infer hour of day from time
df1 = df1.with_columns(((pl.col("time") // 3600) % 24).alias("hour"))

# %%
# Infer title length
df1 = df1.with_columns(pl.col("title").str.len_chars().alias("title_length"))

# %%
# Infer body length
df1 = df1.with_columns(pl.col("text").str.len_chars().alias("body_length"))
df1 = df1.with_columns(pl.col("body_length").fill_null(0))

# %%
# Infer time on newest first page (30 articles are always on first page)
times_published = df1["time"]  # Times of publication

time_on_first_page = []
for i in range(0, len(times_published) - 30):
    time_on_first_page.append(times_published[i + 30] - times_published[i])

time_on_first_page += [None] * 30  # Last 30 articles are undefinetly on first page

time_on_first_page = pl.Series("time_on_first_page", time_on_first_page)
df1 = df1.with_columns(time_on_first_page)

# %%
# Infer average score of posts before a given post on the newest page after 30 minutes
# TODO: There are not exactly 30 posts before each post, after 30 minutes - better approach?
scores = df1["score"]  # Scores of posts on newest page

average_score = []
for i in range(0, len(scores) - 30):
    average_score.append(scores[i : i + 30].median())

average_score += [None] * 30  # No median calculated for last 30 articles
average_score = pl.Series("average_score_of_posts_before", average_score)
df1 = df1.with_columns(average_score)


# %%
# Infer average score of posts on front page after 30 minutes after publication
df1 = df1.with_columns(
    ((pl.col("time") + pl.col("minutes_since_pub") * 60) // 3600).alias(
        "scraped_at_hour"
    )
)

hourly_median_fp_df = (
    df3["score", "scraped_at_hour"]
    .group_by("scraped_at_hour")
    .agg(pl.col("score").median().alias("hourly_median_fp_score"))
)

df1 = df1.join(hourly_median_fp_df, on="scraped_at_hour", how="left")
# %%
# Infer times url has been posted before in the last 12h
df1 = df1.with_columns(
    pl.Series(
        "count_last_12h",
        [
            df1.filter(
                (pl.col("url") == row["url"])
                & (pl.col("time") < row["time"])
                & (pl.col("time") >= (row["time"] - (12 * 3600)))
            ).height
            for row in df1.iter_rows(named=True)
        ],
    )
)

# %%
# Infer times url has been posted before in the last 48h
df1 = df1.with_columns(
    pl.Series(
        "count_last_48h",
        [
            df1.filter(
                (pl.col("url") == row["url"])
                & (pl.col("time") < row["time"])
                & (pl.col("time") >= (row["time"] - (48 * 3600)))
            ).height
            for row in df1.iter_rows(named=True)
        ],
    )
)

# %%
df1 = infer_keywords(df1)

# %% Get topic and document length

conn = sqlite3.connect("../../data/v7/scraped_data.db")
cursor = conn.cursor()
cursor.execute("SELECT id, topic, length FROM topic_webpages")
rows = cursor.fetchall()
conn.close()

ids = [row[0] for row in rows]
topics = [row[1] for row in rows]
document_lengths = [row[2] for row in rows]

topics_df = pl.DataFrame(
    {"id": ids, "topic": topics, "document_length": document_lengths}
)

df1 = df1.join(topics_df, on="id", how="inner")

# %% Infer count of buzzwords
stop_words = set(stopwords.words("english"))
titles = df3["title"].to_list()
title_words = " ".join(titles)
title_words = title_words.lower()
title_words = title_words.split()
title_words = [word for word in title_words if word not in stop_words]

most_successfull_words = [words for words, _ in Counter(title_words).most_common(90)]

df1 = analyse_title(df1, most_successfull_words, "buzzwords", True)


# %%
df1.write_csv("../../data/regression/data.csv")

# %%
