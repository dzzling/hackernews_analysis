import csv
import json
import sqlite3
from bs4 import BeautifulSoup


def save_links(version):
    # Save links to json from csv
    urls_and_ids = {}

    # TODO: Some of the urls went missing, so we need to join the dataframes and check if the url is in the other
    """ 
    df_30min = pl.read_csv(f"data/{version}/30min_data.csv")
    df_240min = pl.read_csv(f"data/{version}/240min_data.csv")

    df = df_30min.join(df_240min, on="id", how="outer") 
    """

    with open(f"data/{version}/30min_data.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls_and_ids[row["id"]] = row["url"]

    with open(f"data/{version}/links.json", "w") as p:
        json.dump(urls_and_ids, p)


def transform_to_plain(DB_NAME, SOURCE_TABLE, DEST_TABLE):

    def extract_relevant_text(html):
        soup = BeautifulSoup(html, "html.parser")

        # TODO: Advance text extraction logic
        text_list = [
            tag.get_text(strip=True)
            for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"])
        ]

        return "\n".join(text_list)  # Combine all extracted text

    # Connect to SQLite
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create destination table if not exists
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DEST_TABLE} (
            id INTEGER,
            plain TEXT
        )
    """
    )
    conn.commit()

    # Fetch data from source table
    cursor.execute(f"SELECT id, html_content FROM {SOURCE_TABLE}")
    rows = cursor.fetchall()

    # Process each row
    data_to_insert = []
    for row in rows:
        id_value, html_content = row
        if html_content:  # Ensure there's HTML content
            text = extract_relevant_text(html_content)

            data_to_insert.append((id_value, text))

    # Insert extracted data into new table
    if data_to_insert:
        cursor.executemany(
            f"INSERT INTO {DEST_TABLE} (id, plain) VALUES (?, ?)", data_to_insert
        )
        conn.commit()

    # Close connection
    conn.close()
