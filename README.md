# Korean spacing errors and subword segmentation (HW2 code)

Code for *"At Which Boundaries Do Korean Pretrained Language Models Break under Spacing Errors?"*
(Jeonghoon Shin, Hanyang University, CSE4057 HW2).

## Environment
- Google Colab, one NVIDIA T4 (16 GB), fp16; Python 3 as provided by Colab
- `pip install -r requirements.txt` (`kiwipiepy==0.24.0` is pinned: the perturbation sites depend on it)
- Exact package versions of the reported runs: `results/env_freeze.txt`

## Data
KLUE-NLI v1.1 (Park et al., 2021; CC BY-SA 4.0). `data/` already contains the converted files and the
evaluation sets. To rebuild them from the official release:
```
git clone --depth 1 https://github.com/KLUE-benchmark/KLUE /tmp/klue
python code/prepare_data.py --klue /tmp/klue --out data
bash code/make_sets.sh        # Set A (k=1), Set B (k=1, k=2); deterministic, seed 0
```
| file | sentences | condition |
|---|---|---|
| `data/setA_k1.jsonl` | 602 | hypotheses admitting JOSA, DEP and EOJEOL; k = 1 |
| `data/setB_k1.jsonl` | 2,961 | hypotheses admitting JOSA and EOJEOL; k = 1 |
| `data/setB_k2.jsonl` | 2,394 | hypotheses admitting JOSA and EOJEOL; k = 2 |

Each row holds the clean hypothesis and one variant per condition: JOSA, DEP (Set A only), EOJEOL,
RANDOM (boundary-agnostic baseline) and NOSPACE.

## Reproducing the tables
Easiest: open `hw2_colab.ipynb` in Colab and run all cells. Or, on any GPU machine:
```
SETS="data/setA_k1.jsonl data/setB_k1.jsonl data/setB_k2.jsonl"
for m in klue/bert-base bert-base-multilingual-cased xlm-roberta-base; do
  for s in 0 1 2; do python code/run_experiment.py --model $m --seed $s --sets $SETS --outdir results; done
done
# replacement for the diverged XLM-R seed 0 (see Seeds)
python code/run_experiment.py --model xlm-roberta-base --seed 3 --sets $SETS --outdir results
for n in setA_k1 setB_k1 setB_k2; do
  python code/token_diff.py --input data/$n.jsonl --models klue/bert-base bert-base-multilingual-cased xlm-roberta-base --out results/tokdiff/$n.csv
done
python code/analyze.py --results results --sets data --tokdiff results/tokdiff --out results/tables
```
| Paper table | Output file | Produced by |
|---|---|---|
| Clean accuracy | `results/tables/table1_clean.*` | `run_experiment.py` → `analyze.py` |
| Drops on Set A | `results/tables/table2_setA_k1.*` | same |
| Pairwise differences (H1) | `results/tables/table3_pairs_setA.*` | same |
| Set B, k ablation | `results/tables/table4_setB.*` | same |
| Segmentation change (H2) | `results/tables/table5_mediation.*` | + `token_diff.py` |

## Seeds
- Fine-tuning: seeds 0, 1, 2 (`--seed`, also used as the data-order seed). A run whose clean dev
  accuracy is at chance level (< 40%; it predicts one class for every input) has diverged; it is
  excluded from all tables and replaced by the next seed (`hw2_colab.ipynb`, Cell 5). In the reported
  runs this happened once: XLM-R seed 0 (33.3%), replaced by seed 3.
- Even with a fixed seed, fp16 training on a GPU is not bit-for-bit deterministic: rerunning
  KLUE-BERT seed 0 changed clean dev accuracy from 82.0% to 82.5%. Conclusions rest on the mean
  over three seeds and on bootstrap CIs.
- Perturbation sampling: seed 0 (`make_sets.sh`)
- Bootstrap: seed 0, B = 2,000 (`analyze.py`)

## Hyperparameters
3 epochs, learning rate 3e-5, batch size 32, warmup ratio 0.1, weight decay 0.01, max length 128, fp16
(identical for all models; see `HP` in `code/run_experiment.py`).

## Files
| path | purpose |
|---|---|
| `code/perturb.py` | boundary-typed spacing perturbation with Kiwi |
| `code/token_diff.py` | segmentation change per tokenizer |
| `code/run_experiment.py` | fine-tune on clean data and evaluate all conditions (no checkpoint saved) |
| `code/analyze.py` | bootstrap statistics and all tables |
| `code/prepare_data.py`, `code/make_sets.sh` | data conversion and evaluation sets |
| `hw2_colab.ipynb` | end-to-end Colab notebook |
