# Imports
import polars as pl
import streamlit as st

df = pl.read_csv("./../../data/v2/front_page_data.csv", ignore_errors=True)
df2 = pl.read_csv("./../../data/v2/30min_data.csv", ignore_errors=True)

# Get more data from 30min scraper
df = df.join(df2, on="id", how="inner")

# Get unique posts
df = df.group_by("id", maintain_order=True).first()

print(f"Number of samples: {df.shape[0]}")

# Exploring overall distributions

tab1, tab2 = st.tabs(["Basic stats", "Distributions"])

# Basic stats

with tab1:

    st.write(f"Number of samples: {df.shape[0]}")

    col3, col4 = st.columns(2)

    with col3:

        average_score = df.select(pl.col("score").mean().alias("average_score"))
        print(average_score)

        st.text("Average score:")
        st.write(average_score)

        average_comments = df.select(
            pl.col("descendants").mean().alias("average_comments")
        )
        print(average_comments)

        st.text("Average comments:")
        st.write(average_comments)

        average_user_post_count = df.select(
            pl.col("user_post_count").mean().alias("average_user_post_count")
        )
        print(average_user_post_count)

        st.text("Average user post count:")
        st.write(average_user_post_count)

        average_minutes_since_pub = df.select(
            pl.col("minutes_since_pub").mean().alias("average_minutes_since_pub")
        )
        print(average_minutes_since_pub)

        st.text("Average minutes since publication:")
        st.write(average_minutes_since_pub)

        average_user_karma = df.select(
            pl.col("user_karma").mean().alias("average_user_karma")
        )
        print(average_user_karma)

        st.text("Average user karma:")
        st.write(average_user_karma)

        average_title_length = df.select(
            pl.col("title").str.len_chars().mean().alias("average_title_length")
        )
        print(average_title_length)

        st.text("Average title length:")
        st.write(average_title_length)

    with col4:

        median_score = df.select(pl.col("score").median().alias("median_score"))
        print(median_score)

        st.text("Median score:")
        st.write(median_score)

        median_comments = df.select(
            pl.col("descendants").median().alias("median_comments")
        )
        print(median_comments)

        st.text("Median comments:")
        st.write(median_comments)

        median_user_post_count = df.select(
            pl.col("user_post_count").median().alias("median_user_post_count")
        )
        print(median_user_post_count)

        st.text("Median user post count:")
        st.write(median_user_post_count)

        median_minutes_since_pub = df.select(
            pl.col("minutes_since_pub").median().alias("median_minutes_since_pub")
        )
        print(median_minutes_since_pub)

        st.text("Median minutes since publication:")
        st.write(median_minutes_since_pub)

        median_user_karma = df.select(
            pl.col("user_karma").median().alias("median_user_karma")
        )
        print(median_user_karma)

        st.text("Median user karma:")
        st.write(median_user_karma)

# Distributions

with tab2:

    col1, col2 = st.columns(2)

    with col1:

        score_distribution = df["score"].hist()
        print(score_distribution)

        st.text("Score distribution:")
        st.dataframe(score_distribution)

        comments_distribution = df["descendants"].hist()
        print(comments_distribution)

        st.text("Comments distribution:")
        st.dataframe(comments_distribution)

        post_count_distribution = df["user_post_count"].hist()
        print(post_count_distribution)

        st.text("User post count distribution:")
        st.dataframe(post_count_distribution)

        minutes_since_pub_distribution = df["minutes_since_pub"].hist()
        print(minutes_since_pub_distribution)

        st.text("Minutes since publication distribution:")
        st.dataframe(minutes_since_pub_distribution)

        user_karma_distribution = df["user_karma"].hist()
        print(user_karma_distribution)

        st.text("User karma distribution:")
        st.dataframe(user_karma_distribution)

    with col2:
        st.text("Score distribution:")
        st.bar_chart(score_distribution, x="breakpoint", y="count")

        st.text("Comments distribution:")
        st.bar_chart(comments_distribution, x="breakpoint", y="count")

        st.text("User post count distribution:")
        st.bar_chart(post_count_distribution, x="breakpoint", y="count")

        st.text("Minutes since publication distribution:")
        st.bar_chart(minutes_since_pub_distribution, x="breakpoint", y="count")

        st.text("User karma distribution:")
        st.bar_chart(user_karma_distribution, x="breakpoint", y="count")
    # %%
