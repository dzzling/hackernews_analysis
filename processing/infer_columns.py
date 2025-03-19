# %%
# Dependencies

import polars as pl

# %%
# Read into dataframe

df1 = pl.read_csv("../data/v6/30min_data.csv", ignore_errors=True)
df2 = pl.read_csv("../data/v6/240min_data.csv", ignore_errors=True)
df3 = pl.read_csv("../data/v6/front_page_data.csv", ignore_errors=True)
df4 = pl.read_csv("../data/v6/second_chance_data.csv", ignore_errors=True)

df1 = df1.join(df2, on="id", how="inner")

""" # %%
# Infer post time on front page
unique_ids = df1["id"].unique()
for unique_id in unique_ids:
    same_posts = df3.filter(pl.col("id") == unique_id)
    first_time, last_time = same_posts[0]["time"], same_posts[-1]["time"] """

# %%
# Infer time scraped
df3 = df3.with_columns(
    (pl.col("time") + (pl.col("minutes_since_pub") * 60)).alias("time_scraped")
)
# Fix this?

# %%
# Infer weekday from time
df1 = df1.with_columns(((((pl.col("time") - 32400) // 86400) + 4) % 7).alias("weekday"))

# %%
# Infer hour from time
df1 = df1.with_columns(((pl.col("time") // 3600) % 24).alias("hour"))

# %%
# Infer title length
df1 = df1.with_columns(pl.col("title").str.len_chars().alias("title_length"))

# %%
# Infer body length
df1 = df1.with_columns(pl.col("text").str.len_chars().alias("body_length"))

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
scores = df1["score"]  # Scores of posts on newest page

average_score = []
for i in range(0, len(scores) - 30):
    average_score.append(scores[i : i + 30].median())

average_score += [None] * 30  # No median calculated for last 30 articles
average_score = pl.Series("average_score", average_score)
df1 = df1.with_columns(average_score)


# %%
# Infer average score of posts on front page after 30 minutes after publication
scores = df1["score"]  # Scores of posts on newest page
