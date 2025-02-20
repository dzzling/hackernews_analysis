# %%
# Dependencies
import polars as pl
import altair as alt
import math

alt.data_transformers.enable("vegafusion")

# %%
# Read into dataframe

df = pl.read_csv("./../../data/data.csv", ignore_errors=True)

# %%
# Score distribution
fig = alt.Chart(df).mark_bar().encode(x="score", y="count()")
fig

# %%
# Average score by weekday
## Calculate time - 9h to get US time
## 0: Sunday, 1:Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday, 6:Saturday
df = df.with_columns(((((pl.col("time") - 32400) // 86400) + 4) % 7).alias("weekday"))
fig = alt.Chart(df).mark_bar().encode(x="weekday:Q", y="mean(score):Q")
fig

# %%
# Average score by hour
df = df.with_columns(((pl.col("time") // 3600) % 24).alias("hour"))
fig = alt.Chart(df).mark_bar().encode(x="hour:Q", y="mean(score):Q")
fig

# %%
# Median score by hour
fig = alt.Chart(df).mark_bar().encode(x="hour:Q", y="median(score):Q")
fig

# %%
# Score by comment count
fig = alt.Chart(df).mark_point().encode(x="descendants:Q", y="score:Q", size="count()")
fig

# %%
# Counts of posts by user
## Not score related
user_df = (df.to_series(0)).value_counts(name="entry_count")

breaks = [0, 5, 10, 15, 20, 25, 30]
binned = user_df.with_columns(posts_in_period=pl.col("entry_count").cut(breaks))
binned = binned.with_columns(pl.col("posts_in_period").cast(pl.String))
binned.head()

fig = alt.Chart(binned).mark_bar().encode(x="posts_in_period:N", y="count()")
fig

# %%
# Scores and user karma
fig = alt.Chart(df).mark_point().encode(x="user_karma:Q", y="score:Q")
fig

# %%
# Scores and post count by user
breaks = [0, 5, 10, 50, 100, 500, 1000, 50000, 100000]
binned = df.with_columns(post_count=pl.col("user_post_count").cut(breaks))
binned = binned.with_columns(pl.col("post_count").cast(pl.String))
fig = (
    alt.Chart(binned)
    .mark_point()
    .encode(x="post_count:N", y="median(score):Q", size="count()")
)
fig

# %%
# Median score by user
