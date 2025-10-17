#!/usr/bin/env python3
"""
ubuntu_requests.py
Fetch an image from a user-provided URL and save it in ./Fetched_Images/
Implements graceful error handling and respectful HTTP checks.
"""

import os
import sys
import requests
from urllib.parse import urlparse, unquote
from pathlib import Path
from datetime import datetime

FETCH_DIR = Path("Fetched_Images")
TIMEOUT = 10  # seconds


def make_fetch_dir():
    FETCH_DIR.mkdir(parents=True, exist_ok=True)


def extract_filename_from_url(url):
    """
    Try to extract a sensible filename from the URL path.
    If not possible, generate a timestamped filename with extension guessed from content-type later.
    """
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    if name:
        return name
    # no filename in URL path
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"fetched_image_{timestamp}"


def guess_extension_from_content_type(content_type):
    if not content_type:
        return ""
    content_type = content_type.split(";")[0].strip().lower()
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    return mapping.get(content_type, "")


def download_image(url):
    headers = {
        "User-Agent": "UbuntuRequestsBot/1.0 (+https://example.com/) - respectful fetcher"
    }
    try:
        with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as resp:
            resp.raise_for_status()  # raise HTTPError for 4xx/5xx
            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                print(f"⚠️ The URL does not look like an image (Content-Type: {content_type}). Aborting.")
                return None

            # Determine filename
            base_name = extract_filename_from_url(url)
            ext = Path(base_name).suffix
            if not ext:
                ext = guess_extension_from_content_type(content_type) or ".img"

            # Ensure unique filename to avoid overwriting
            final_name = base_name if Path(base_name).suffix else base_name + ext
            final_path = FETCH_DIR / final_name
            counter = 1
            while final_path.exists():
                final_path = FETCH_DIR / f"{Path(final_name).stem}_{counter}{Path(final_name).suffix}"
                counter += 1

            # Save in binary chunks
            with open(final_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return final_path

    except requests.exceptions.Timeout:
        print("❌ Error: The request timed out. Try again later or check your connection.")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server. Check the URL or your internet connection.")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error happened: {e}")
    except PermissionError:
        print("❌ Permission error: Can't write file to disk in the current directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    return None


def main():
    print("Ubuntu Image Fetcher — I am because we are 🌍")
    url = input("Enter the image URL to fetch: ").strip()
    if not url:
        print("No URL entered. Exiting.")
        return

    # Basic URL validation
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        print("❌ Only http:// or https:// URLs are supported. Please enter a valid URL.")
        return

    make_fetch_dir()
    saved_path = download_image(url)
    if saved_path:
        print(f"✅ Image saved to: {saved_path.resolve()}")
    else:
        print("⚠️ Image was not saved due to previous errors.")


if __name__ == "__main__":
    # Ensure requests exists
    try:
        import requests  # already imported above, but double-check for user clarity
    except ImportError:
        print("The 'requests' library is required. Install it with: pip install requests")
        sys.exit(1)

    main()
