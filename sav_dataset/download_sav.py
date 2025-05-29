import requests
from pathlib import Path

LINKS_FILE = "sav_links.txt"
OUT_DIR    = Path("sav_downloads")
OUT_DIR.mkdir(exist_ok=True)

with open(LINKS_FILE, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

for line in lines:
    # split on tab; skip malformed lines
    parts = line.split("\t")
    if len(parts) != 2:
        print(f"Skipping malformed line: {line}")
        continue
    fname, url = parts
    out_path = OUT_DIR / fname
    if out_path.exists():
        print(f"{fname} already exists, skipping.")
        continue

    print(f"Downloading {fname} …")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as out_f:
                for chunk in r.iter_content(chunk_size=8192):
                    out_f.write(chunk)
    except Exception as e:
        print(f"  → Error downloading {fname}: {e}")
