"""Print how each tokenizer segments one hypothesis before and after each perturbation
(the example table in the paper). Output: results/tokenization_example.json

    python code/tokenization_example.py
"""
import json
import os

from transformers import AutoTokenizer

CLEAN = "정부는 내년 예산을 늘릴 것이라고 밝혔다"
PERT = {"EOJEOL": "정부는내년 예산을 늘릴 것이라고 밝혔다",
        "JOSA": "정부는 내년 예산 을 늘릴 것이라고 밝혔다",
        "DEP": "정부는 내년 예산을 늘릴것이라고 밝혔다"}
out = {}
for m in ["klue/bert-base", "bert-base-multilingual-cased", "xlm-roberta-base"]:
    tok = AutoTokenizer.from_pretrained(m)
    out[m] = {"CLEAN": tok.tokenize(CLEAN), **{k: tok.tokenize(v) for k, v in PERT.items()}}
    print("==", m)
    for k, v in out[m].items():
        print(k.ljust(6), v)
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/tokenization_example.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
