import os
import sys
import argparse
import json
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Rag.nlp_utils import preprocess, ngrams


def read_texts(paths: List[str]) -> List[str]:
    texts = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(('.txt', '.md')):
                        with open(os.path.join(root, f), 'r', encoding='utf-8', errors='ignore') as fh:
                            texts.append(fh.read())
        else:
            with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                texts.append(fh.read())
    return texts


def main():
    ap = argparse.ArgumentParser(description='Build medical corpus and basic statistics')
    ap.add_argument('--inputs', nargs='+', required=True, help='Files or directories of texts')
    ap.add_argument('--out', default='corpus_stats.json', help='Output JSON path')
    args = ap.parse_args()

    texts = read_texts(args.inputs)
    token_counts = {}
    bigram_counts = {}
    trigram_counts = {}

    for txt in texts:
        res = preprocess(txt)
        for t in res['normalized']:
            token_counts[t] = token_counts.get(t, 0) + 1
        for bg in ngrams(res['normalized'], 2):
            bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
        for tg in ngrams(res['normalized'], 3):
            trigram_counts[tg] = trigram_counts.get(tg, 0) + 1

    # Keep top terms
    def top_k(d, k=100):
        return [[*k_, v] if isinstance(k_, tuple) else [k_, v] for k_, v in sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]]

    stats = {
        'total_documents': len(texts),
        'top_tokens': top_k(token_counts, 200),
        'top_bigrams': top_k(bigram_counts, 100),
        'top_trigrams': top_k(trigram_counts, 100)
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"Corpus stats written to {args.out}")


if __name__ == '__main__':
    main()


