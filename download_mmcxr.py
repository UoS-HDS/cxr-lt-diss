import asyncio
import time
from pathlib import Path
import os

import httpx
from bs4 import BeautifulSoup


BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/"
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
    save_to: set[str] = set(),
):
    """"""
    print(f"Crawling: {url}...")
    try:
        response = await client.get(url, auth=(USERNAME, PASSWORD), headers=HEADERS)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return save_to

    soup = BeautifulSoup(response.text, "html.parser")

    tasks = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue

        full_url = clean_url(url, href)

        if is_file(href):
            print(f"Found file: {full_url}")
            save_to.add(full_url)
        elif is_subdir(href):
            tasks.append(crawl_directory(client, full_url, save_to=save_to))

    if len(tasks) > 0:
        await asyncio.gather(*tasks, return_exceptions=True)
    else:
        return save_to


async def download_image(client: httpx.AsyncClient, url: str):
    """"""
    try:
        response = await client.get(url)
        response.raise_for_status()

        file_path = DATA_PATH / url.replace(BASE_URL, "")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("wb") as f:
            f.write(response.content)

        print(f"Finished downloading {file_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")


async def main():
    """main function"""
    saved_urls = Path("urls.txt")
    limits = httpx.Limits(max_connections=50)
    img_urls = set()

    if not saved_urls.exists():
        start = time.time()
        async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
            img_urls = await crawl_directory(client, BASE_URL, save_to=img_urls)

        print(f"Time taken to crawl: {time.time() - start:.2f} seconds")

        print(f"Discovered {len(img_urls)} files.")
        with open("urls.txt", "w") as f:
            for u in sorted(img_urls):
                f.write(u + "\n")
    else:
        with saved_urls.open("r") as f:
            img_urls = (line.strip() for line in f)

    return
    start = time.time()
    async with httpx.AsyncClient(
        auth=(USERNAME, PASSWORD), headers=HEADERS, limits=limits, timeout=30.0
    ) as client:
        tasks = []
        for url in img_urls:
            tasks.append(download_image(client, url))
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
