"""Boundary-typed Korean word-spacing perturbation.

Each sentence is analysed with Kiwi, and every candidate edit site is assigned
one boundary type:

  JOSA   : a particle (tag J*) attached to its host  -> INSERT a space   (밥을 -> 밥 을)
  DEP    : a space before a dependent noun (NNB)      -> DELETE the space (먹을 수 -> 먹을수)
  EOJEOL : any other space between two eojeols        -> DELETE the space (밥을 먹었다 -> 밥을먹었다)

Baseline:
  RANDOM : k sites drawn uniformly from all JOSA/DEP/EOJEOL sites, ignoring the boundary
           type (a boundary-agnostic spacing perturbation in the spirit of random space
           deletion/insertion used in prior robustness suites)

Optional:
  AUX    : a space before an auxiliary predicate (VX) -> DELETE the space.
           Korean orthography *permits* attaching most auxiliaries (보조 용언 붙여 씀 허용),
           so AUX is not a true error and is kept separate from DEP.

To compare types fairly, the same number of edits (k) is applied for every type,
and only sentences that have at least k sites of *every* compared type are kept
(the "matched subset").

Usage:
  python perturb.py --input clean.jsonl --field title --k 1 --seed 0 --out perturbed.jsonl
  python perturb.py --demo
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

from kiwipiepy import Kiwi

TYPES = ("JOSA", "DEP", "EOJEOL")


@dataclass(frozen=True)
class Site:
    kind: str   # JOSA / DEP / EOJEOL / AUX
    pos: int    # character offset in the original string
    op: str     # "insert" (a space at pos) or "delete" (the space at pos)


_kiwi: Kiwi | None = None


def kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def find_sites(text: str) -> list[Site]:
    toks = kiwi().tokenize(text)
    sites: list[Site] = []
    seen_spaces: set[int] = set()
    for prev, cur in zip(toks, toks[1:]):
        prev_end = prev.start + prev.len
        gap = text[prev_end:cur.start]
        if prev.tag.startswith("S") or cur.tag.startswith("S"):
            continue  # skip punctuation / symbols / foreign tokens
        if gap == "":
            # attached morphemes: only particles become JOSA sites
            if cur.tag.startswith("J"):
                sites.append(Site("JOSA", cur.start, "insert"))
        elif gap == " ":
            space_pos = prev_end
            if space_pos in seen_spaces:
                continue
            seen_spaces.add(space_pos)
            if cur.tag == "NNB":
                kind = "DEP"
            elif cur.tag == "VX":
                kind = "AUX"
            else:
                kind = "EOJEOL"
            sites.append(Site(kind, space_pos, "delete"))
    return sites


def apply(text: str, chosen: list[Site]) -> str:
    # apply from right to left so earlier offsets stay valid
    out = text
    for s in sorted(chosen, key=lambda s: s.pos, reverse=True):
        if s.op == "insert":
            out = out[: s.pos] + " " + out[s.pos:]
        else:
            assert out[s.pos] == " ", (out, s)
            out = out[: s.pos] + out[s.pos + 1:]
    return out


def perturb(text: str, kind: str, k: int, rng: random.Random) -> tuple[str, list[Site]] | None:
    """kind is one of TYPES, or RANDOM: k sites drawn from all JOSA/DEP/EOJEOL sites
    regardless of boundary type (the boundary-agnostic baseline)."""
    sites = find_sites(text)
    cands = [s for s in sites if (s.kind in TYPES if kind == "RANDOM" else s.kind == kind)]
    if len(cands) < k:
        return None
    chosen = rng.sample(cands, k)
    return apply(text, chosen), chosen


def remove_all_spaces(text: str) -> str:
    """Upper-bound condition: every space deleted."""
    return text.replace(" ", "")


def process(rows: list[dict], field: str, k: int, seed: int, types=TYPES) -> list[dict]:
    """Keep only sentences that admit >= k sites of every type in `types` (the matched set);
    for each, build one variant per type, the RANDOM baseline, and NOSPACE."""
    out = []
    for i, row in enumerate(rows):
        text = row[field]
        rng = random.Random(f"{seed}-{i}")
        variants = {}
        for t in tuple(types) + ("RANDOM",):
            r = perturb(text, t, k, rng)
            if r is None:
                break
            variants[t] = {"text": r[0], "edits": [s.pos for s in r[1]]}
        else:  # all types available -> matched subset
            variants["NOSPACE"] = {"text": remove_all_spaces(text), "edits": []}
            out.append({**row, "clean": text, "variants": variants})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--field", default="title")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--types", default=",".join(TYPES),
                    help="comma-separated types every kept sentence must admit (e.g. JOSA,EOJEOL)")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    types = tuple(t for t in a.types.split(",") if t)

    if a.demo:
        rows = [{"title": "나는 오늘 학교에서 밥을 먹을 수 있었다"},
                {"title": "정부는 내년 예산을 늘릴 것이라고 밝혔다"},
                {"title": "그는 책을 읽어 보았다"}]
        for r in process(rows, "title", a.k, a.seed):
            print("CLEAN  ", r["clean"])
            for t, v in r["variants"].items():
                print(f"{t:8s}", v["text"])
            print()
        n_all = len(rows)
        print(f"matched subset: {len(process(rows, 'title', a.k, a.seed))}/{n_all}")
        return

    rows = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    res = process(rows, a.field, a.k, a.seed, types)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(res)}/{len(rows)} sentences (matched on {','.join(types)}, k={a.k})")


if __name__ == "__main__":
    main()
