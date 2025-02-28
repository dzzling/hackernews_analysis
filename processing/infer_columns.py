# Dependencies

import polars as pl

# Read into dataframe

df = pl.read_csv("./../../data/v1/data.csv", ignore_errors=True)

# Infer weekday from time
df = df.with_columns(((((pl.col("time") - 32400) // 86400) + 4) % 7).alias("weekday"))

# Infer hour from time
df = df.with_columns(((pl.col("time") // 3600) % 24).alias("hour"))

# Infer title length
df = df.with_columns(pl.col("title").str.len_chars().alias("title_length"))

# Infer body length
df = df.with_columns(pl.col("text").str.len_chars().alias("body_length"))

### ... continued
