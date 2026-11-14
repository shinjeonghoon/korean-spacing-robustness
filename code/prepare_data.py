"""Convert the official KLUE-NLI v1.1 JSON files to JSONL used by every other script.

    git clone --depth 1 https://github.com/KLUE-benchmark/KLUE /tmp/klue
    python code/prepare_data.py --klue /tmp/klue --out data

Labels: entailment = 0, neutral = 1, contradiction = 2.
KLUE is released under CC BY-SA 4.0 (Park et al., 2021).
"""
import argparse
import json
import os

LABEL = {"entailment": 0, "neutral": 1, "contradiction": 2}

ap = argparse.ArgumentParser()
ap.add_argument("--klue", required=True)
ap.add_argument("--out", default="data")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for split in ("train", "dev"):
    src = os.path.join(a.klue, "klue_benchmark", "klue-nli-v1.1", f"klue-nli-v1.1_{split}.json")
    rows = json.load(open(src, encoding="utf-8"))
    with open(os.path.join(a.out, f"{split}.jsonl"), "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps({"guid": x["guid"], "premise": x["premise"], "hypothesis": x["hypothesis"],
                                "label": LABEL[x["gold_label"]]}, ensure_ascii=False) + "\n")
    print(split, len(rows))
