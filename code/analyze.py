"""Aggregate results over models and seeds and produce every table of the paper.

    python analyze.py --results results --sets data --tokdiff results/tokdiff --out results/tables

Inputs
  results/<model>_s<seed>/train_log.json, preds_<set>.csv   (run_experiment.py)
  results/tokdiff/<set>.csv                                  (token_diff.py)
  data/<set>.jsonl                                           (perturb.py)
Outputs (CSV + LaTeX + Markdown) in --out
  table1_clean.*        clean dev accuracy per model (mean ± sd over seeds)
  table2_setA_k1.*      accuracy drop per condition on Set A, with 95% CIs
  table3_pairs_setA.*   pairwise differences of drops (H1 decision)
  table4_setB.*         drops on Set B for k = 1, 2 (ablation)
  table5_mediation.*    segmentation change vs. drop (H2): median split + logistic regression
  examples_random.csv   randomly sampled perturbed examples with predictions (incl. failures)
  manual_check_100.csv  100 random perturbed hypotheses to label for meaning change

Statistics: a per-sentence drop is averaged over the three seeds, then a paired
nonparametric bootstrap over sentences (B = 2,000, seed 0) gives 95% percentile CIs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

B = 2000
RNG_SEED = 0
# A fine-tuning run whose clean dev accuracy is at chance level (3 balanced classes = 33.3%)
# has diverged (it predicts one class for every input). Such runs are excluded and reported;
# the run with the next seed replaces it (see README).
DIVERGED_BELOW = 0.40


def run_dirs(results):
    """All finished runs, split into (kept, diverged) by the clean dev accuracy rule."""
    kept, bad = [], []
    for d in sorted(glob.glob(os.path.join(results, "*_s*"))):
        f = os.path.join(d, "train_log.json")
        if not os.path.exists(f):
            continue
        (bad if json.load(open(f))["clean_dev_acc"] < DIVERGED_BELOW else kept).append(d)
    return kept, bad
MODEL_ORDER = ["bert-base", "bert-base-multilingual-cased", "xlm-roberta-base"]
MODEL_NAME = {"bert-base": "KLUE-BERT", "bert-base-multilingual-cased": "mBERT", "xlm-roberta-base": "XLM-R"}


def boot(x: np.ndarray, rng, b=B):
    idx = rng.integers(0, len(x), size=(b, len(x)))
    s = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def fmt(m, lo, hi, pct=True):
    k = 100 if pct else 1
    return f"{m*k:.1f} [{lo*k:.1f}, {hi*k:.1f}]"


def load_preds(results, set_name):
    """-> {model: DataFrame indexed by idx, columns = conditions, values = mean correctness over seeds}"""
    out = {}
    for d in run_dirs(results)[0]:
        f = os.path.join(d, f"preds_{set_name}.csv")
        if not os.path.exists(f):
            continue
        model = os.path.basename(d).rsplit("_s", 1)[0]
        df = pd.read_csv(f).set_index("idx").drop(columns=["guid"])
        out.setdefault(model, []).append(df)
    return {m: sum(v) / len(v) for m, v in out.items()}, {m: len(v) for m, v in out.items()}


def save(df, out, name, caption):
    df.to_csv(os.path.join(out, name + ".csv"))
    with open(os.path.join(out, name + ".md"), "w") as f:
        f.write(df.to_markdown() + "\n")
    with open(os.path.join(out, name + ".tex"), "w") as f:
        f.write(df.to_latex(caption=caption, label="tab:" + name, escape=True))
    print(f"\n== {name} ==\n{df.to_markdown()}")


def order(models):
    return [m for m in MODEL_ORDER if m in models] + [m for m in models if m not in MODEL_ORDER]


def table1(results, out):
    kept, bad = run_dirs(results)
    rows, seeds, diverged = {}, {}, {}
    for d in kept + bad:
        j = json.load(open(os.path.join(d, "train_log.json")))
        m = j["model"].split("/")[-1]
        if d in bad:
            diverged.setdefault(m, []).append(f"s{j['seed']} ({100*j['clean_dev_acc']:.1f}%)")
            continue
        rows.setdefault(m, []).append((j["clean_dev_acc"], j["train_minutes"]))
        seeds.setdefault(m, []).append(j["seed"])
    df = pd.DataFrame({MODEL_NAME.get(m, m): {
        "Clean dev acc (%)": f"{100*np.mean([a for a, _ in v]):.1f} ± {100*np.std([a for a, _ in v]):.1f}",
        "Seeds used": ",".join(map(str, sorted(seeds[m]))), "Diverged (excluded)": "; ".join(diverged.get(m, [])) or "–",
        "Train min / run": f"{np.mean([t for _, t in v]):.1f}"} for m, v in
        ((m, rows[m]) for m in order(rows))}).T
    save(df, out, "table1_clean", "Clean KLUE-NLI dev accuracy (mean ± sd over seeds).")


def drop_table(preds, conds, rng):
    rows = {}
    for m in order(preds):
        df = preds[m]
        r = {"Clean acc": fmt(*boot(df["CLEAN"].to_numpy(), rng))}
        for c in conds:
            if c in df:
                r[c] = fmt(*boot((df["CLEAN"] - df[c]).to_numpy(), rng))
        rows[MODEL_NAME.get(m, m)] = r
    return pd.DataFrame(rows).T


def table_pairs(preds, rng):
    pairs = [("EOJEOL", "DEP"), ("DEP", "JOSA"), ("EOJEOL", "JOSA"),
             ("JOSA", "RANDOM"), ("DEP", "RANDOM"), ("EOJEOL", "RANDOM")]
    rows = {}
    for m in order(preds):
        df = preds[m]
        r = {}
        for a, b in pairs:
            # Drop(a) - Drop(b) = Acc(b) - Acc(a), paired over the same sentences
            m_, lo, hi = boot((df[b] - df[a]).to_numpy(), rng)
            r[f"{a} − {b}"] = fmt(m_, lo, hi) + (" *" if lo > 0 or hi < 0 else "")
        rows[MODEL_NAME.get(m, m)] = r
    return pd.DataFrame(rows).T


def mediation(preds, tokdiff, rng):
    """H2. Per model and type: (a) drop difference high- vs low-change (median split of edit_norm);
    (b) logistic regression of a per-sentence flip on standardized edit_norm, controlling for
    delta_len and type, with a sentence-level bootstrap CI for the edit_norm coefficient."""
    import statsmodels.api as sm
    rows = {}
    for m in order(preds):
        td = tokdiff[tokdiff["model"].str.split("/").str[-1] == m]
        df = preds[m]
        r = {}
        frames = []
        for t in ["JOSA", "EOJEOL", "RANDOM"]:
            g = td[td["type"] == t].set_index("idx")
            d = (df["CLEAN"] - df[t]).loc[g.index]
            hi = g["edit_norm"] > g["edit_norm"].median()
            x = rng.integers(0, len(g), size=(B, len(g)))
            dd, hh = d.to_numpy(), hi.to_numpy()
            diffs = [dd[i][hh[i]].mean() - dd[i][~hh[i]].mean() for i in x]
            r[f"{t}: high − low"] = fmt(dd[hh].mean() - dd[~hh].mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
            frames.append(pd.DataFrame({"flip": (d > 0).astype(int).to_numpy(), "edit_norm": g["edit_norm"].to_numpy(),
                                        "delta_len": g["delta_len"].to_numpy(), "type": t}))
        X = pd.concat(frames, ignore_index=True)
        X["z"] = (X["edit_norm"] - X["edit_norm"].mean()) / X["edit_norm"].std()
        design = pd.get_dummies(X[["z", "delta_len", "type"]], columns=["type"], drop_first=True, dtype=float)
        design = sm.add_constant(design)
        if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
            design = design.drop(columns=["delta_len"])      # collinear with edit_norm: drop the control
            r["controls"] = "type"
        else:
            r["controls"] = "type, delta_len"
        fit = lambda yy, XX: sm.Logit(yy, XX).fit(disp=0, method="bfgs", maxiter=200).params["z"]
        coef = fit(X["flip"], design)
        n = len(X)
        bs = []
        for _ in range(500):
            i = rng.integers(0, n, n)
            try:
                bs.append(fit(X["flip"].iloc[i], design.iloc[i]))
            except Exception:
                pass
        r["logit coef of edit_norm (z)"] = f"{coef:.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"
        rows[MODEL_NAME.get(m, m)] = r
    return pd.DataFrame(rows).T


def matched_change(preds, tokdiff, rng):
    """Is the drop explained by how much the segmentation changes, or by which boundary is hit?
    JOSA and EOJEOL edits are pooled, split into quartiles of edit_norm (per tokenizer), and the
    drops of the two types are compared within each quartile, i.e. at a similar segmentation change."""
    rows = {}
    for m in order(preds):
        td = tokdiff[tokdiff["model"].str.split("/").str[-1] == m]
        df = preds[m]
        parts = []
        for t in ["JOSA", "EOJEOL"]:
            g = td[td["type"] == t].set_index("idx")
            parts.append(pd.DataFrame({"type": t, "e": g["edit_norm"], "drop": (df["CLEAN"] - df[t]).loc[g.index]}))
        X = pd.concat(parts)
        X["q"] = pd.qcut(X["e"].rank(method="first"), 4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])
        for q in X["q"].cat.categories:
            r = {}
            for t in ["JOSA", "EOJEOL"]:
                x = X[(X["q"] == q) & (X["type"] == t)]["drop"].to_numpy()
                r[f"{t} n"] = len(x)
                r[f"{t} drop"] = fmt(*boot(x, rng))
            rows[(MODEL_NAME.get(m, m), q)] = r
    return pd.DataFrame(rows).T


def eojeol_nounnoun(preds, sets_dir, rng):
    """EOJEOL edits that join two nouns (often an acceptable compound, e.g. 무기 체계 -> 무기체계)
    versus all other EOJEOL edits, Set B (k=1). Uses Kiwi tags of the clean hypothesis."""
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    rows = [json.loads(l) for l in open(os.path.join(sets_dir, "setB_k1.jsonl"), encoding="utf-8")]
    nn = []
    for r in rows:
        pos = r["variants"]["EOJEOL"]["edits"][0]
        toks = kiwi.tokenize(r["clean"])
        L = [t for t in toks if t.start + t.len == pos]
        R = [t for t in toks if t.start == pos + 1]
        nn.append(bool(L and R and L[0].tag.startswith("NN") and R[0].tag.startswith("NN")))
    nn = np.array(nn)
    out = {}
    for m in order(preds):
        d = (preds[m]["CLEAN"] - preds[m]["EOJEOL"]).to_numpy()
        out[MODEL_NAME.get(m, m)] = {f"noun+noun (n={nn.sum()})": fmt(*boot(d[nn], rng)),
                                     f"other (n={(~nn).sum()})": fmt(*boot(d[~nn], rng))}
    return pd.DataFrame(out).T


def examples(results, sets_dir, out, rng, n_per_type=5):
    rows = [json.loads(l) for l in open(os.path.join(sets_dir, "setA_k1.jsonl"), encoding="utf-8")]
    preds = {os.path.basename(d).rsplit("_s", 1)[0]: pd.read_csv(os.path.join(d, "preds_setA_k1.csv")).set_index("idx")
             for d in [min((x for x in run_dirs(results)[0] if os.path.basename(x).rsplit("_s", 1)[0] == m),
                           key=lambda x: int(x.rsplit("_s", 1)[1]))
                       for m in {os.path.basename(x).rsplit("_s", 1)[0] for x in run_dirs(results)[0]}]}
    recs = []
    for t in ["JOSA", "DEP", "EOJEOL"]:
        for i in rng.choice(len(rows), n_per_type, replace=False):
            r = rows[i]
            rec = {"type": t, "idx": int(i), "premise": r["premise"], "clean": r["clean"], "perturbed": r["variants"][t]["text"], "label": r["label"]}
            for m, p in preds.items():
                rec[f"{MODEL_NAME.get(m, m)} clean/pert"] = f"{p.loc[i, 'CLEAN']}/{p.loc[i, t]}"
            recs.append(rec)
    pd.DataFrame(recs).to_csv(os.path.join(out, "examples_random.csv"), index=False, encoding="utf-8-sig")
    # control (ii): 100 random perturbed hypotheses for a manual meaning-change check
    pool = [(i, t) for i in range(len(rows)) for t in ["JOSA", "DEP", "EOJEOL"]]
    pick = rng.choice(len(pool), 100, replace=False)
    pd.DataFrame([{"idx": pool[j][0], "type": pool[j][1], "clean": rows[pool[j][0]]["clean"],
                   "perturbed": rows[pool[j][0]]["variants"][pool[j][1]]["text"], "meaning_changed (0/1)": ""} for j in pick]
                 ).to_csv(os.path.join(out, "manual_check_100.csv"), index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--sets", default="data")
    ap.add_argument("--tokdiff", default="results/tokdiff")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    table1(a.results, a.out)
    pA, nA = load_preds(a.results, "setA_k1")
    print("seeds per model:", nA)
    save(drop_table(pA, ["JOSA", "DEP", "EOJEOL", "RANDOM", "NOSPACE"], rng), a.out, "table2_setA_k1",
         "Accuracy drop (percentage points, 95\\% CI) on Set A (k=1).")
    save(table_pairs(pA, rng), a.out, "table3_pairs_setA",
         "Pairwise differences of drops on Set A; * marks CIs excluding zero.")
    parts = []
    for k in (1, 2):
        pB, _ = load_preds(a.results, f"setB_k{k}")
        t = drop_table(pB, ["JOSA", "EOJEOL", "RANDOM", "NOSPACE"], rng)
        t.index = [f"{i} (k={k})" for i in t.index]
        parts.append(t)
    save(pd.concat(parts), a.out, "table4_setB", "Accuracy drop on Set B for k=1 and k=2.")
    td_path = os.path.join(a.tokdiff, "setB_k1.csv")
    if os.path.exists(td_path):
        pB1, _ = load_preds(a.results, "setB_k1")
        save(mediation(pB1, pd.read_csv(td_path), rng), a.out, "table5_mediation",
             "Segmentation change and accuracy drop on Set B (k=1).")
        save(matched_change(pB1, pd.read_csv(td_path), rng), a.out, "table7_matched_change",
             "Drops of JOSA and EOJEOL within quartiles of segmentation change (Set B, k=1).")
    pB1, _ = load_preds(a.results, "setB_k1")
    save(eojeol_nounnoun(pB1, a.sets, np.random.default_rng(RNG_SEED + 2)), a.out, "table8_eojeol_nounnoun",
         "EOJEOL drop for noun+noun joins vs. other joins (Set B, k=1).")
    # own generator, so the sampled examples do not depend on how many bootstrap draws ran before
    examples(a.results, a.sets, a.out, np.random.default_rng(RNG_SEED + 1))


if __name__ == "__main__":
    main()
