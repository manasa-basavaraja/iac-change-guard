# iac-change-guard

Reads unified diffs for terraform, kubernetes yaml, and github actions. Builds a small tree when the yaml or hcl parses, otherwise uses a few hand rolled signals from the added lines. Runs the diff text through DistilBERT, mixes that with the tree vector using attention, then a three way head benign vs misconfig vs malicious.

Examples under `examples/` are tiny. `train_demo.py` only fits on those five files so do not read much into the numbers.

## layout

Collector and tokenizer live under `src/iac_guard/`. Ast module, distilbert wrapper, fusion, classifier, `model.py`, `pipeline.py` are there too. `scripts/detect.py` and `train_demo.py` are the entrypoints.

## setup

Use 64 bit python 3.10 or newer. On windows `py -3.10` is what I use.

```
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

First run pulls distilbert from huggingface, a few hundred mb.

If torch fails to import with dll errors, install the vc++ redist x64 from microsoft then reinstall torch.

## run

Random weights until you train:

```
python scripts/detect.py --diff examples/terraform_wide_open.diff
```

Train on the bundled diffs then point detect at the checkpoint:

```
python scripts/train_demo.py --epochs 3 --out checkpoints/demo.pt
python scripts/detect.py --diff examples/gha_exfil_style.diff --ckpt checkpoints/demo.pt
```

Tests:

```
python -m pytest tests/ -q
```

Defaults live in `configs/default.yaml` including the 0.72 cutoff on the malicious class if you want an alert style gate.

## license

MIT in `LICENSE`.
