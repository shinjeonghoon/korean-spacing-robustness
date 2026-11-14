"""Does a dependent noun survive as its own subword after the preceding space is deleted?

For every Set A sentence, the DEP variant deletes the space before a dependent noun (NNB).
For each tokenizer we check whether the noun (e.g. 것, 수) still appears as a whole token
piece (ignoring the "##" / "▁" markers) after the edit, and compare the segmentation change
of DEP with JOSA and EOJEOL on the same sentences. With --preds, the accuracy drop is split
by whether the noun survived.

    python code/dep_tokens.py --set data/setA_k1.jsonl \
        --models klue/bert-base bert-base-multilingual-cased xlm-roberta-base \
        --preds results --out results/tables/table6_dep_tokens
    python code/dep_tokens.py --demo
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from kiwipiepy import Kiwi

from token_diff import diff

MARKERS = ("##", "▁")


def strip(piece: str) -> str:
    for m in MARKERS:
        if piece.startswith(m):
            piece = piece[len(m):]
    return piece


def dep_noun(kiwi: Kiwi, clean: str, pos: int) -> str | None:
    """The NNB that follows the deleted space at character offset `pos` of the clean text."""
    for t in kiwi.tokenize(clean):
        if t.start == pos + 1 and t.tag == "NNB":
            return t.form
    return None


def survives(tok, text: str, noun: str) -> bool:
    return any(strip(p) == noun for p in tok(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="data/setA_k1.jsonl")
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--preds", default=None, help="results dir with <model>_s<seed>/preds_setA_k1.csv")
    ap.add_argument("--out", default="results/tables/table6_dep_tokens")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    kiwi = Kiwi()

    if a.demo:
        def wp(s):  # toy WordPiece: whole words are split into syllables
            out = []
            for w in s.split(" "):
                out += [w[0]] + ["##" + c for c in w[1:]]
            return out
        clean = "밥을 먹을 수 있었다"
        pos = clean.index(" 수")
        pert = clean[:pos] + clean[pos + 1:]
        noun = dep_noun(kiwi, clean, pos)
        print(noun, wp(pert), survives(wp, pert, noun), diff(wp, clean, pert))
        return

    import pandas as pd
    from transformers import AutoTokenizer
    rows = [json.loads(l) for l in open(a.set, encoding="utf-8")]
    nouns = [dep_noun(kiwi, r["clean"], r["variants"]["DEP"]["edits"][0]) for r in rows]
    recs = []
    for name in a.models:
        tok = AutoTokenizer.from_pretrained(name).tokenize
        for i, (r, n) in enumerate(zip(rows, nouns)):
            rec = {"model": name, "idx": i, "noun": n,
                   "survives": survives(tok, r["variants"]["DEP"]["text"], n) if n else None}
            for t in ("JOSA", "DEP", "EOJEOL"):
                rec[f"edit_norm_{t}"] = diff(tok, r["clean"], r["variants"][t]["text"])["edit_norm"]
            recs.append(rec)
    df = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out + "_per_sentence.csv", index=False, encoding="utf-8-sig")

    summary = df.groupby("model").agg(
        n=("idx", "size"), noun_survives=("survives", "mean"),
        edit_JOSA=("edit_norm_JOSA", "mean"), edit_DEP=("edit_norm_DEP", "mean"),
        edit_EOJEOL=("edit_norm_EOJEOL", "mean"))
    if a.preds:
        for name in a.models:
            tag = name.split("/")[-1]
            from analyze import run_dirs          # same divergence rule as the tables
            fs = [os.path.join(d, "preds_setA_k1.csv") for d in run_dirs(a.preds)[0]
                  if os.path.basename(d).rsplit("_s", 1)[0] == tag]
            if not fs:
                continue
            p = sum(pd.read_csv(f).set_index("idx")[["CLEAN", "DEP"]] for f in fs) / len(fs)
            d = (p["CLEAN"] - p["DEP"])
            s = df[df["model"] == name].set_index("idx")["survives"].astype(bool)
            summary.loc[name, "drop_if_survives"] = d[s].mean()
            summary.loc[name, "drop_if_split"] = d[~s].mean() if (~s).any() else float("nan")
            summary.loc[name, "n_split"] = int((~s).sum())
    summary.to_csv(a.out + ".csv")
    with open(a.out + ".md", "w") as f:
        f.write(summary.to_markdown(floatfmt=".3f") + "\n")
    top = df[df["model"] == a.models[0]]["noun"].value_counts().head(10)
    print(summary.to_markdown(floatfmt=".3f"))
    print("\nmost frequent dependent nouns:", dict(top))


if __name__ == "__main__":
    main()
