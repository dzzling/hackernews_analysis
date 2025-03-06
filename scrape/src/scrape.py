import requests
import sqlite3
import json
import time
from bs4 import BeautifulSoup
from tools import save_links, transform_to_plain

# Database setup
DB_NAME = "data/scraped_data.db"
TABLE_NAME = "webpages"


def create_database():
    """Create SQLite database and table if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            html_content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    conn.commit()
    conn.close()


def scrape_and_store(cursor, id, url):
    """Scrapes a webpage and stores the content in the database."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        html_content = soup.prettify()  # Store formatted HTML

        cursor.execute(
            f"""
            INSERT OR IGNORE INTO {TABLE_NAME} (id, url, html_content)
            VALUES (?, ?, ?)
        """,
            (id, url, html_content),
        )

        print(f"Successfully scraped and stored: {url}")
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")


def main():
    version = "v4"

    save_links(version)
    time.sleep(5)  # Wait for the links to be saved
    create_database()

    # List of URLs to scrape
    with open(f"data/{version}/links.json", "r") as f:
        URLS = json.load(f)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for id, url in URLS.items():
        if url == "":
            continue
        scrape_and_store(cursor, id, url)
        conn.commit()

    conn.close()

    transform_to_plain()


if __name__ == "__main__":
    main()
