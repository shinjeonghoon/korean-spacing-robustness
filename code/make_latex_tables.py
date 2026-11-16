"""Turn results/tables/*.csv into compact booktabs tables for the paper (\\input-able).

    python code/make_latex_tables.py --tables results/tables --out paper/tables
"""
import argparse
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tables", default="results/tables")
ap.add_argument("--out", default="paper/tables")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T = lambda n: pd.read_csv(os.path.join(a.tables, n + ".csv"), index_col=0)


def cell(s):
    """'4.0 [2.2, 5.8] *' -> '4.0 {\\scriptsize[2.2, 5.8]}' (bold if marked significant)."""
    s = str(s)
    star = s.endswith("*")
    s = s.rstrip(" *")
    if "[" not in s:
        return s
    m, ci = s.split(" [")
    out = f"{m} {{\\scriptsize[{ci}}}"
    return f"\\textbf{{{m}}} {{\\scriptsize[{ci}}}" if star else out


def write(name, body):
    with open(os.path.join(a.out, name + ".tex"), "w") as f:
        f.write(body)


# Table: clean accuracy
t = T("table1_clean")
rows = "\n".join(f"{m} & {r['Clean dev acc (%)'].replace('±', '$\\pm$')} & {r['Seeds used']} & {r['Train min / run']} \\\\" for m, r in t.iterrows())
write("clean", f"""\\begin{{tabular}}{{lccc}}
\\toprule
Model & Clean acc. (\\%) & Seeds & Min/run \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
""")

# Table: Set A drops
t = T("table2_setA_k1")
cols = ["JOSA", "DEP", "EOJEOL", "RANDOM", "NOSPACE"]
rows = "\n".join(f"{m} & " + " & ".join(cell(r[c]) for c in cols) + " \\\\" for m, r in t.iterrows())
write("setA", f"""\\begin{{tabular}}{{l{'c' * len(cols)}}}
\\toprule
Model & \\textsc{{Josa}} & \\textsc{{Dep}} & \\textsc{{Eojeol}} & \\textsc{{Random}} & \\textsc{{NoSpace}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
""")

# Table: Set B, k = 1 and 2
t = T("table4_setB")
cols = ["JOSA", "EOJEOL", "RANDOM", "NOSPACE"]
lines = []
for k in (1, 2):
    lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{k = {k}}}}} \\\\")
    for m, r in t[t.index.str.endswith(f"(k={k})")].iterrows():
        lines.append(m.replace(f" (k={k})", "") + " & " + " & ".join(cell(r[c]) for c in cols) + " \\\\")
body = "\n".join(lines)
write("setB", f"""\\begin{{tabular}}{{lcccc}}
\\toprule
Model & \\textsc{{Josa}} & \\textsc{{Eojeol}} & \\textsc{{Random}} & \\textsc{{NoSpace}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
""")

# Table: pairwise differences on Set A (appendix)
t = T("table3_pairs_setA")
cols = list(t.columns)
hdr = " & ".join(c.replace("−", "$-$") for c in cols)
rows = "\n".join(f"{m} & " + " & ".join(cell(r[c]) for c in cols) + " \\\\" for m, r in t.iterrows())
write("pairs", f"""\\begin{{tabular}}{{l{'c' * len(cols)}}}
\\toprule
Model & {hdr} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
""")

# Table: segmentation change (H2)
t = T("table5_mediation")
rows = "\n".join(f"{m} & {cell(r['logit coef of edit_norm (z)'])} & {cell(r['EOJEOL: high − low'])} & {cell(r['RANDOM: high − low'])} & {cell(r['JOSA: high − low'])} \\\\" for m, r in t.iterrows())
write("mediation", f"""\\begin{{tabular}}{{lcccc}}
\\toprule
 & Logit coef. & \\multicolumn{{3}}{{c}}{{Drop: high $-$ low change (pp)}} \\\\
\\cmidrule(lr){{3-5}}
Model & (per SD) & \\textsc{{Eojeol}} & \\textsc{{Random}} & \\textsc{{Josa}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
""")

# Table: dependent nouns
t = T("table6_dep_tokens")
name = {"klue/bert-base": "KLUE-BERT", "bert-base-multilingual-cased": "mBERT", "xlm-roberta-base": "XLM-R"}
order = ["klue/bert-base", "bert-base-multilingual-cased", "xlm-roberta-base"]
rows = "\n".join(f"{name[m]} & {100*t.loc[m,'noun_survives']:.1f} & {t.loc[m,'edit_DEP']:.3f} & {t.loc[m,'edit_EOJEOL']:.3f} & {100*t.loc[m,'drop_if_survives']:.1f} & {100*t.loc[m,'drop_if_split']:.1f} ({int(t.loc[m,'n_split'])}) \\\\" for m in order)
write("dep", f"""\\begin{{tabular}}{{lccccc}}
\\toprule
 & Noun kept & \\multicolumn{{2}}{{c}}{{Seg.\\ change}} & \\multicolumn{{2}}{{c}}{{\\textsc{{Dep}} drop (pp)}} \\\\
\\cmidrule(lr){{3-4}}\\cmidrule(lr){{5-6}}
Model & (\\%) & \\textsc{{Dep}} & \\textsc{{Eojeol}} & kept & split ($n$) \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
""")
print("wrote", sorted(os.listdir(a.out)))

# Table: randomly sampled examples (appendix); predictions of the seed-0 (first valid) run of each model
ex = pd.read_csv(os.path.join(a.tables, "examples_random.csv"))
LAB = {0: "E", 1: "N", 2: "C"}
pc = [c for c in ex.columns if c.endswith("clean/pert")]
order_m = [m for m in ("KLUE-BERT", "mBERT", "XLM-R") if any(c.startswith(m) for c in pc)]
lines = []
for _, r in ex.iterrows():
    preds = " ".join(r[f"{m} clean/pert"] for m in order_m)
    lines.append(f"\\textsc{{{r['type'].title()}}} & {r['clean']} & {r['perturbed']} & {LAB[int(r['label'])]} & {preds} \\\\")
write("examples", "\\begin{tabular}{l p{0.33\\textwidth} p{0.33\\textwidth} c c}\n\\toprule\n"
      "Type & Clean hypothesis & Perturbed hypothesis & Gold & " + "/".join(order_m) + " \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# Table: EOJEOL noun+noun joins vs. other joins
t = T("table8_eojeol_nounnoun")
c1, c2 = t.columns
rows = "\n".join(f"{m} & {cell(r[c1])} & {cell(r[c2])} \\\\" for m, r in t.iterrows())
write("nounnoun", f"""\\begin{{tabular}}{{lcc}}
\\toprule
Model & Noun+noun join ($n={c1.split('=')[1].rstrip(')')}$) & Other joins ($n={c2.split('=')[1].rstrip(')')}$) \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
""")
