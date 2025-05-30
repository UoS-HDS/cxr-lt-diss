import asyncio
import time
from pathlib import Path
import os
from datetime import datetime

import httpx
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm


BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/"
DATA_PATH = Path(__file__).parent / "mimic-cxr-jpg/2.0.0/"
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
if not USERNAME or not PASSWORD:
    print("USERNAME and PASSWORD env variables must be set")
    os._exit(1)
HEADERS = {
    "User-Agent": "Wget/1.21.1",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

DATA_PATH.mkdir(parents=True, exist_ok=True)


def is_file(url_text):
    return "." in url_text and not url_text.endswith("/")


def is_subdir(url_text):
    return url_text.endswith("/") and not url_text.startswith("../")


def clean_url(base, href):
    return str(httpx.URL(base).join(href))


async def crawl_directory(
    client: httpx.AsyncClient,
    url: str,
    save_to: set[str],
    depth: int = 0,
):
    """Crawl a directory and collect all file URLs"""
    indent = "  " * depth
    print(f"{indent}[{datetime.now().strftime('%H:%M:%S')}] Crawling: {url}")

    try:
        response = await asyncio.wait_for(
            client.get(url, follow_redirects=True, auth=(USERNAME, PASSWORD)),
            timeout=20.0,
        )
        print(f"{indent}[DEBUG] {url} → {response.status_code}")
        response.raise_for_status()

    except asyncio.TimeoutError:
        print(f"{indent}[TIMEOUT] {url}")
        return
    except Exception as e:
        print(f"{indent}[ERROR] {url} -> {type(e).__name__}: {e}")
        return

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"{indent}[ERROR] Failed to parse HTML for {url}: {e}")
        return

    # Collect files and subdirectories
    files_found = 0
    subdirs = []

    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue

        full_url = clean_url(url, href)

        if is_file(href):
            save_to.add(full_url)
            files_found += 1
            if files_found <= 3:  # Show first few files
                print(f"{indent}  Found file: {full_url}")

        elif is_subdir(href):
            subdirs.append(full_url)

    if files_found > 3:
        print(f"{indent}  Found {files_found} files total in this directory")

    if subdirs:
        print(f"{indent}  Found {len(subdirs)} subdirectories")

    print(f"{indent}  Total files discovered so far: {len(save_to)}")

    # Process subdirectories recursively
    for subdir_url in subdirs:
        await crawl_directory(client, subdir_url, save_to, depth + 1)


async def download_image(client: httpx.AsyncClient, url: str):
    """Download a single image"""
    try:
        response = await asyncio.wait_for(client.get(url), timeout=30.0)
        response.raise_for_status()

        file_path = DATA_PATH / url.replace(BASE_URL, "")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("wb") as f:
            f.write(response.content)

        print(f"Downloaded {file_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")


async def main():
    """Main function"""
    SAVED_URLS = Path("urls.txt")
    img_urls = set()

    if not SAVED_URLS.exists():
        print("Starting crawl to discover image URLs...")
        start = time.time()

        async with httpx.AsyncClient(
            auth=(USERNAME, PASSWORD),
            headers=HEADERS,
            timeout=30.0,
            cookies=httpx.Cookies(),
        ) as client:
            try:
                await crawl_directory(client, BASE_URL, img_urls)

            except Exception as e:
                print(f"[ERROR] Crawling failed: {e}")
                raise

        print(f"\nCrawling completed in {time.time() - start:.2f} seconds")
        print(f"Discovered {len(img_urls)} files.")

        if img_urls:
            print("Saving URLs to urls.txt...")
            with open("urls.txt", "w") as f:
                for u in sorted(img_urls):
                    f.write(u + "\n")
            print("URLs saved successfully!")
        else:
            print("WARNING: No URLs found - check for errors above")
            return
    else:
        print("Loading URLs from existing urls.txt file...")
        with SAVED_URLS.open("r") as f:
            img_urls = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(img_urls)} URLs from file")

    # Early return to skip downloads (as in original script)
    return

    print("\nStarting downloads...")
    start = time.time()
    async with httpx.AsyncClient(
        auth=(USERNAME, PASSWORD), headers=HEADERS, timeout=30.0
    ) as client:
        tasks = []
        for url in img_urls:
            tasks.append(download_image(client, url))

        # Download with progress bar
        for f in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Downloading dataset",
        ):
            await f

    print(f"Downloads completed in {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
        os._exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        os._exit(2)
