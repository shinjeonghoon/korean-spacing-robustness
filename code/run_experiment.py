"""Fine-tune one encoder on clean KLUE-NLI and evaluate it on clean and spacing-perturbed
dev sets, in one run (no checkpoint is kept, so a Colab session never fills the disk).

    python run_experiment.py --model klue/bert-base --seed 0 \
        --train data/train.jsonl --dev data/dev.jsonl \
        --sets data/setA_k1.jsonl data/setB_k1.jsonl data/setB_k2.jsonl \
        --outdir results

Writes, under --outdir/<model-tag>_s<seed>/:
  train_log.json        hyperparameters, wall-clock time, clean dev accuracy (all 3,000 pairs)
  preds_<set>.csv       per-example correctness (1/0) for CLEAN and every perturbation condition
Only the hypothesis is perturbed; the premise always stays clean.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import time

import numpy as np
import torch
from datasets import Dataset
import transformers
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

MAX_LEN = 128
HP = dict(epochs=3, lr=3e-5, batch=32, warmup_ratio=0.1, weight_decay=0.01)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


@torch.no_grad()
def predict(model, tok, premises, hyps, bs=128):
    out = []
    for i in range(0, len(premises), bs):
        enc = tok(premises[i:i + bs], hyps[i:i + bs], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(model.device)
        out += model(**enc).logits.argmax(-1).tolist()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--sets", nargs="+", default=[])
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    tag = a.model.split("/")[-1] + f"_s{a.seed}"
    out = os.path.join(a.outdir, tag)
    os.makedirs(out, exist_ok=True)
    set_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    def enc(b):
        return tok(b["premise"], b["hypothesis"], truncation=True, max_length=MAX_LEN)
    train = Dataset.from_list(read_jsonl(a.train)).map(enc, batched=True)
    dev_rows = read_jsonl(a.dev)

    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=3)
    # Warmup is given as a step count computed from the ratio: `warmup_ratio` was removed in
    # transformers v5, and arguments unknown to the installed version are dropped.
    total_steps = math.ceil(len(train) / HP["batch"]) * HP["epochs"]
    kwargs = dict(
        output_dir=os.path.join("/tmp", tag), seed=a.seed, data_seed=a.seed,
        num_train_epochs=HP["epochs"], learning_rate=HP["lr"],
        per_device_train_batch_size=HP["batch"], warmup_steps=int(HP["warmup_ratio"] * total_steps),
        weight_decay=HP["weight_decay"], fp16=torch.cuda.is_available(),
        save_strategy="no", logging_steps=200, report_to=[])
    accepted = inspect.signature(TrainingArguments.__init__).parameters
    dropped = [k for k in kwargs if k not in accepted]
    if dropped:
        print("TrainingArguments: ignoring unsupported arguments", dropped)
    args = TrainingArguments(**{k: v for k, v in kwargs.items() if k in accepted})
    trainer = Trainer(model=model, args=args, train_dataset=train,
                      data_collator=DataCollatorWithPadding(tok))
    t0 = time.time()
    trainer.train()
    train_min = (time.time() - t0) / 60
    model.eval()

    gold = np.array([r["label"] for r in dev_rows])
    dev_acc = float((predict(model, tok, [r["premise"] for r in dev_rows],
                             [r["hypothesis"] for r in dev_rows]) == gold).mean())
    print(f"[{tag}] clean dev acc = {dev_acc:.4f}  ({train_min:.1f} min training)")

    for sp in a.sets:
        rows = read_jsonl(sp)
        prem = [r["premise"] for r in rows]
        y = np.array([r["label"] for r in rows])
        conds = {"CLEAN": [r["clean"] for r in rows]}
        for c in rows[0]["variants"]:
            conds[c] = [r["variants"][c]["text"] for r in rows]
        correct = {c: (predict(model, tok, prem, h) == y).astype(int) for c, h in conds.items()}
        name = os.path.splitext(os.path.basename(sp))[0]
        with open(os.path.join(out, f"preds_{name}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "guid"] + list(correct))
            for i, r in enumerate(rows):
                w.writerow([i, r["guid"]] + [int(correct[c][i]) for c in correct])
        print(f"[{tag}] {name}: " + "  ".join(f"{c}={v.mean():.3f}" for c, v in correct.items()))

    json.dump({"model": a.model, "seed": a.seed, "hyperparameters": HP, "max_len": MAX_LEN,
               "warmup_steps": kwargs["warmup_steps"], "total_steps": total_steps,
               "transformers": transformers.__version__, "torch": torch.__version__,
               "train_minutes": round(train_min, 2), "clean_dev_acc": dev_acc,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
              open(os.path.join(out, "train_log.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
