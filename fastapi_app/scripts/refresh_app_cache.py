"""
Standalone CLI script to scrape Google Play Store for autism & speech therapy apps,
formatting them and refreshing fastapi_app/files/app_cache.json.

Run manually or via cron worker:
    python fastapi_app/scripts/refresh_app_cache.py
"""

import json
import logging
import re
from pathlib import Path
from google_play_scraper import search, app as play_store_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
APP_CACHE_PATH = BASE_DIR / "files/app_cache.json"

def clean_html(text: str) -> str:
    return re.sub('<[^<]+?>', '', text)

def refresh_app_cache(query: str = "autism speech therapy special education", max_results: int = 50):
    log.info(f"Querying Google Play Store for '{query}'...")
    try:
        search_results = search(query, lang="en", country="us")
    except Exception as e:
        log.error(f"Failed to query Google Play: {e}")
        return

    top_apps = search_results[:max_results]
    app_data = []

    for item in top_apps:
        app_id = item.get("appId")
        try:
            details = play_store_app(app_id, lang="en", country="us")
            genre = details.get("genre", "")
            if genre in ["Education", "Medical", "Parenting"]:
                app_data.append({
                    "App_Name": details.get("title", "Unknown App"),
                    "Category": genre,
                    "Rating": round(float(details.get("score") or 0), 2),
                    "Price": "Free" if details.get("free") else "Paid",
                    "Description": clean_html(details.get("description", ""))[:600],
                    "App_Link": details.get("url", f"https://play.google.com/store/apps/details?id={app_id}")
                })
                log.info(f"  + Added app: {details.get('title')}")
        except Exception as e:
            log.warning(f"  - Failed fetching details for {app_id}: {e}")
            continue

    if app_data:
        APP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(APP_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(app_data, f, indent=2, ensure_ascii=False)
        log.info(f"? Successfully wrote {len(app_data)} apps to {APP_CACHE_PATH}")
    else:
        log.warning("No matching apps found. app_cache.json was not updated.")

if __name__ == "__main__":
    refresh_app_cache()
