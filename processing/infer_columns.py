# Dependencies

import polars as pl

# Read into dataframe

df1 = pl.read_csv("./../../data/v2/30min_data.csv", ignore_errors=True)
df2 = pl.read_csv("./../../data/v2/240min_data.csv", ignore_errors=True)
df3 = pl.read_csv("./../../data/v2/front_page_data.csv", ignore_errors=True)
df4 = pl.read_csv("./../../data/v2/second_chance_pool.csv", ignore_errors=True)

# Infer weekday from time
df1 = df1.with_columns(((((pl.col("time") - 32400) // 86400) + 4) % 7).alias("weekday"))

# Infer hour from time
df1 = df1.with_columns(((pl.col("time") // 3600) % 24).alias("hour"))

# Infer title length
df1 = df1.with_columns(pl.col("title").str.len_chars().alias("title_length"))

# Infer body length
df1 = df1.with_columns(pl.col("text").str.len_chars().alias("body_length"))
