# %%
# Dependencies
import polars as pl
import altair as alt

alt.data_transformers.enable("vegafusion")

# %% Read into dataframe

df = pl.read_csv("./../../data/v1/data.csv", ignore_errors=True)

print(f"Number of samples: {df.shape[0]}")

# Exploring overall distributions

# %% Basic stats
top_5_score = df.sort("score", descending=True).head(5)
print(top_5_score)
top_5_comments = df.fill_null(0).sort("descendants", descending=True).head(5)
print(top_5_comments)
# %% Score distribution
fig = (
    alt.Chart(df, title="Score distribution").mark_bar().encode(x="score", y="count()")
)
fig

# %% Average score by weekday
## Calculate time - 9h to get US time
## 0: Sunday, 1:Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday, 6:Saturday
df = df.with_columns(((((pl.col("time") - 32400) // 86400) + 4) % 7).alias("weekday"))
fig = (
    alt.Chart(df, title="Average score by weekday")
    .mark_bar()
    .encode(x="weekday:Q", y="mean(score):Q")
)
fig

# %% Average score by hour
df = df.with_columns(((pl.col("time") // 3600) % 24).alias("hour"))
fig = (
    alt.Chart(df, title="Average score by hour")
    .mark_bar()
    .encode(x="hour:Q", y="mean(score):Q")
)
fig

# %% Median score by hour
fig = (
    alt.Chart(df, title="Median score by hour")
    .mark_bar()
    .encode(x="hour:Q", y="median(score):Q")
)
fig

# %% Score by comment count
fig = (
    alt.Chart(df, title="Score by comment count")
    .mark_point()
    .encode(x="descendants:Q", y="score:Q", size="count()")
)
fig

# %% Counts of posts by user in period
## Not score related
user_df = df.get_column("by").value_counts(name="entry_count")

breaks = [0, 5, 10, 15, 20, 25, 30]
binned = user_df.with_columns(
    posts_in_period=pl.col("entry_count").cut(
        breaks,
        labels=[
            "<0",
            "a 0-5",
            "b 5-10",
            "c 10-15",
            "d 15-20",
            "e 20-25",
            "f 25-30",
            "g 30+",
        ],
    )
)
binned = binned.with_columns(pl.col("posts_in_period").cast(pl.String))
binned.head()

fig = (
    alt.Chart(binned, title="Counts of posts by user in period")
    .mark_bar()
    .encode(x="posts_in_period:N", y="count()")
)
fig

# %% Scores and user karma (full)
fig = (
    alt.Chart(df, title="Scores and user karma (full)")
    .mark_point()
    .encode(x="user_karma:Q", y="score:Q")
)
fig

# %% Scores and user karma (median)
breaks = [0, 50, 100, 500, 1000, 5000, 10000, 100000]
## Assign karma bucket to each user
binned = df.with_columns(
    user_karma=pl.col("user_karma")
    .cut(
        breaks,
        labels=[
            "<0",
            "a 0-50",
            "b 50-100",
            "c 100-500",
            "d 500-1000",
            "e 1000-5000",
            "f 5000-10000",
            "g 10000-100000",
            "h 100000+",
        ],
    )
    .cast(pl.String)
)
## Counts of posts by karma bucket
user_count = binned.get_column("user_karma").value_counts(name="entry_count")
## Group by karma bucket and join with counts
binned = (
    binned.group_by("user_karma")
    .agg(pl.col("score").median().alias("median_score"))
    .join(user_count, on="user_karma")
)
fig = (
    alt.Chart(binned, title="Scores and user karma (median)")
    .mark_point()
    .encode(x="user_karma:N", y="median_score:Q", size="entry_count")
)
fig

# %% Median scores and median post count by user

## Get median score by user, take first state of post count
binned = df.group_by("by").agg(
    pl.col("score").median().alias("median_score"),
    pl.col("user_post_count").first().alias("Total post count"),
)

## Assign total post count bucket to each user
breaks = [0, 5, 10, 50, 100, 500, 1000, 50000, 100000]
binned = binned.with_columns(
    total_post_count=pl.col("Total post count").cut(
        breaks,
        labels=[
            "a 0-",
            "b 0-5",
            "c 5-10",
            "d 10-50",
            "e 50-100",
            "f 100-500",
            "g 500-1000",
            "h 1000-50000",
            "i 50000-100000",
            "j 100000+",
        ],
    )
)
binned = binned.with_columns(pl.col("total_post_count").cast(pl.String))

## Register how many users are in each bucket
counts = (binned.get_column("total_post_count")).value_counts(name="entry_count")

## Groub by bucket and get median score
binned = binned.group_by("total_post_count").agg(pl.col("median_score").median())

## Join the counts to the binned dataframe
binned = binned.join(counts, on="total_post_count")

## Visualize
fig = (
    alt.Chart(binned, title="Median scores and median post count by user")
    .mark_point()
    .encode(
        x=alt.X("total_post_count:N", title="Total post count"),
        y=alt.Y("median_score:Q", title="Median score of user in bucket"),
        size=alt.Size("entry_count", legend=alt.Legend(title="Users in bucket")),
    )
)
fig

# %% Mean scores and mean post count by user

## Get mean score by user, take first state of post count
binned = df.group_by("by").agg(
    pl.col("score").mean().alias("mean_score"),
    pl.col("user_post_count").first().alias("Total post count"),
)

## Assign total post count bucket to each user
breaks = [0, 5, 10, 50, 100, 500, 1000, 50000, 100000]
binned = binned.with_columns(
    total_post_count=pl.col("Total post count").cut(
        breaks,
        labels=[
            "a 0-",
            "b 0-5",
            "c 5-10",
            "d 10-50",
            "e 50-100",
            "f 100-500",
            "g 500-1000",
            "h 1000-50000",
            "i 50000-100000",
            "j 100000+",
        ],
    )
)
binned = binned.with_columns(pl.col("total_post_count").cast(pl.String))

## Register how many users are in each bucket
counts = (binned.get_column("total_post_count")).value_counts(name="entry_count")

## Groub by bucket and get mean score
binned = binned.group_by("total_post_count").agg(pl.col("mean_score").mean())

## Join the counts to the binned dataframe
binned = binned.join(counts, on="total_post_count")

## Visualize
fig = (
    alt.Chart(binned, title="Mean scores and mean post count by user")
    .mark_point()
    .encode(
        x=alt.X("total_post_count:N", title="Total post count"),
        y=alt.Y("mean_score:Q", title="Mean score of user in bucket"),
        size=alt.Size("entry_count", legend=alt.Legend(title="Users in bucket")),
    )
)
fig
