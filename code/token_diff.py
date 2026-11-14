"""Measure how much a spacing perturbation changes the subword segmentation.

For every (clean, perturbed) pair and every tokenizer it reports:
  len_clean, len_pert  : number of subword tokens
  delta_len            : len_pert - len_clean        (length confound)
  edit                 : token-level Levenshtein distance
  edit_norm            : edit / max(len_clean, len_pert)   (candidate mediator)

The surface edit (one space added or removed) is identical across tokenizers,
so differences in `edit_norm` reflect the tokenizer, not the perturbation.

Usage (Colab, needs internet for the Hugging Face hub):
  python token_diff.py --input perturbed.jsonl \
      --models klue/bert-base bert-base-multilingual-cased xlm-roberta-base \
      --out token_diff.csv
  python token_diff.py --demo      # offline sanity check with a character tokenizer
"""
from __future__ import annotations

import argparse
import csv
import json


def levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def diff(tok, clean: str, pert: str) -> dict:
    a, b = tok(clean), tok(pert)
    e = levenshtein(a, b)
    return {"len_clean": len(a), "len_pert": len(b), "delta_len": len(b) - len(a),
            "edit": e, "edit_norm": e / max(len(a), len(b), 1)}


def load_tokenizers(names: list[str]):
    from transformers import AutoTokenizer
    return {n: AutoTokenizer.from_pretrained(n).tokenize for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--out")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.demo:
        # characters with a word-initial marker, mimicking SentencePiece's "▁"
        def char_tok(s):
            out = []
            for w in s.split(" "):
                out += ["▁" + w[0]] + list(w[1:]) if w else []
            return out
        for clean, pert in [("밥을 먹을 수 있었다", "밥 을 먹을 수 있었다"),
                            ("밥을 먹을 수 있었다", "밥을 먹을수 있었다")]:
            print(clean, "->", pert, diff(char_tok, clean, pert))
        assert levenshtein(list("kitten"), list("sitting")) == 3
        print("ok")
        return

    toks = load_tokenizers(a.models)
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = None
        for i, r in enumerate(rows):
            for vtype, v in r["variants"].items():
                for name, tok in toks.items():
                    rec = {"idx": i, "type": vtype, "model": name, **diff(tok, r["clean"], v["text"])}
                    if w is None:
                        w = csv.DictWriter(f, fieldnames=list(rec))
                        w.writeheader()
                    w.writerow(rec)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
