#!/bin/bash
# Build the evaluation sets from data/dev.jsonl (hypothesis perturbed, seed 0).
#   Set A: sentences that admit JOSA, DEP and EOJEOL (k = 1)          -> H1 comparison
#   Set B: sentences that admit JOSA and EOJEOL (k = 1 and k = 2)     -> larger set, k ablation
set -e
cd "$(dirname "$0")/.."
python code/perturb.py --input data/dev.jsonl --field hypothesis --types JOSA,DEP,EOJEOL --k 1 --seed 0 --out data/setA_k1.jsonl
python code/perturb.py --input data/dev.jsonl --field hypothesis --types JOSA,EOJEOL --k 1 --seed 0 --out data/setB_k1.jsonl
python code/perturb.py --input data/dev.jsonl --field hypothesis --types JOSA,EOJEOL --k 2 --seed 0 --out data/setB_k2.jsonl
