import os
import sys
import argparse
import json
import time
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Rag.nlp_utils import preprocess


SOURCES = [
    # Add plain text URLs that can be fetched without auth; placeholder list
    # Users can extend this with specific open-licensed medical content.
    'https://raw.githubusercontent.com/yoavg/word-embeddings-for-nlp/master/data/text8',
]


def fetch_url(url: str) -> str:
    import requests
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def main():
    ap = argparse.ArgumentParser(description='Ingest open texts into a processed corpus')
    ap.add_argument('--out_dir', default='open_corpus', help='Directory to write processed texts')
    ap.add_argument('--extra_url', action='append', help='Additional text URL(s) to include')
    args = ap.parse_args()

    urls = list(SOURCES)
    if args.extra_url:
        urls.extend(args.extra_url)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []

    for url in urls:
        try:
            text = fetch_url(url)
            res = preprocess(text)
            ts = int(time.time())
            base = f"doc_{ts}_{len(manifest)}"
            raw_path = os.path.join(args.out_dir, base + '_raw.txt')
            proc_path = os.path.join(args.out_dir, base + '_proc.json')
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(text)
            with open(proc_path, 'w', encoding='utf-8') as f:
                json.dump(res, f)
            manifest.append({'url': url, 'raw': raw_path, 'processed': proc_path})
            print(f"Ingested {url}")
        except Exception as e:
            print(f"Failed to ingest {url}: {e}")

    with open(os.path.join(args.out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest with {len(manifest)} items")


if __name__ == '__main__':
    main()


