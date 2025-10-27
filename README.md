# Model — Milestone 3 README

This repo fine-tunes a Qwen LLM to **predict Steam game prices** from metadata and review stats. The workflow centers on three scripts:

- `preprocess_data.py` — build training/validation JSONL + numeric features  
- `finetune.py` — fine-tune a causal LLM (LoRA/QLoRA optional)  
- `infer_model.py` — run single/batch inference, compute MAE on batch

---

## Quickstart

```bash
# Clone
git clone https://github.com/indie-launch/Model.git
cd Model

# (Recommended) Docker
chmod +x docker-shell.sh
./docker-shell.sh  # opens a shell with the venv activated

# Or: Jupyter
./docker-shell.sh jupyter lab --LabApp.token='' --ip=0.0.0.0 --no-browser
```

---

## Environment (Docker)

We provide an `uv`-based **Dockerfile** and **docker-shell.sh**:

```bash
# Build (first run auto-builds too)
docker build -t indie-model-uv -f Dockerfile .

# Enter the dev container (repo mounted at /app; venv auto-activated)
./docker-shell.sh

# GPU (auto-detected). Ensure NVIDIA Container Toolkit is installed.
nvidia-smi  # on host
```

> The script mounts your Hugging Face cache and maps your UID/GID so files aren’t owned by root.

---

## 1) Data Preprocessing — `preprocess_data.py`

**Purpose:** Convert Steam JSONL dumps into:
- `qwen_train.jsonl` / `qwen_val.jsonl` for supervised fine-tuning  
- `features.csv` (numeric baseline) + `id_map.csv` + a small `readme.txt`  

**Inputs (JSONL)** expected in repo:
- `steam-app-list.jsonl`, `steam-app-reviews.jsonl`, `steam-raw-app-data.jsonl` (Steam store objects)

**Usage:**
```bash
./docker-shell.sh python preprocess_data.py   --app-list steam-app-list.jsonl   --reviews steam-app-reviews.jsonl   --raw steam-raw-app-data.jsonl   --out-dir steam_out   --min-total-reviews 10   --seed 42   --val-ratio 0.1
```
This writes `qwen_train.jsonl`, `qwen_val.jsonl`, `features.csv`, and `id_map.csv` under `steam_out/`.

---

## 2) Fine-tuning — `finetune.py`

**Goal:** Train a Qwen model to emit **ONLY** a 2-decimal USD price string (e.g., `12.99`). Works full-precision/bf16 or with QLoRA (if CUDA+bitsandbytes).

### Recommended baseline (1.5B)
```bash
./docker-shell.sh python finetune.py   --model Qwen/Qwen2.5-1.5B-Instruct   --train_jsonl steam_out/qwen_train.jsonl   --val_jsonl   steam_out/qwen_val.jsonl   --output_dir  models/qwen-price-sft-1p5b   --epochs 3 --lr 2e-4 --batch_size 4 --grad_accum 4 --max_len 1024
```

### Optional QLoRA (needs CUDA + bitsandbytes)
```bash
./docker-shell.sh python finetune.py   --model Qwen/Qwen2.5-1.5B-Instruct   --train_jsonl steam_out/qwen_train.jsonl   --val_jsonl   steam_out/qwen_val.jsonl   --output_dir  models/qwen-price-sft-1p5b-qlora   --epochs 3 --lr 2e-4 --batch_size 4 --grad_accum 4 --max_len 1024   --use_qlora
```
Examples and flags align with the script’s header docstring.

---

## 3) Inference — `infer_model.py`

Supports **single prompt** or **batch JSONL**. If you trained LoRA adapters, pass `--base_model` and point `--model_path` to the adapter dir. The script parses a numeric price from the model’s output and, in batch mode, prints **MAE** when ground truth is present.

### A) Single text
```bash
./docker-shell.sh python infer_model.py   --model_path models/qwen-price-sft-1p5b   --text_input "$(cat <<'TXT'
Name: Example Game
Genres: Action, Adventure
Categories: Single-player
Developers: Example Studios
Publishers: Example Publishing
Platforms: windows, mac
Is Free: False
Release Year: 2022
Review Summary: Very Positive; Positive: 1234; Negative: 123; Total: 1357
Short Description: Fast-paced adventure...
TXT
)"
```

### B) Batch JSONL → predictions + MAE
```bash
./docker-shell.sh python infer_model.py   --model_path models/qwen-price-sft-1p5b   --input_jsonl steam_out/qwen_val.jsonl   --out predictions.jsonl
```
- For **LoRA adapters**:
```bash
./docker-shell.sh python infer_model.py   --model_path models/qwen-price-sft-1p5b-qlora   --base_model Qwen/Qwen2.5-1.5B-Instruct   --input_jsonl steam_out/qwen_val.jsonl   --out predictions.jsonl
```
Argument behavior and MAE calculation are implemented in the script.

---

## Repo Layout

```
├── preprocess_data.py      # build qwen_train/val JSONL + features
├── finetune.py             # SFT with optional QLoRA (PEFT/bitsandbytes)
├── infer_model.py          # single/batch inference, MAE
├── steam-app-*.jsonl       # raw inputs (provided in repo)
├── Dockerfile              # uv-based Python 3.11 slim
├── docker-shell.sh         # helper: build/run with venv & caches
└── steam_out/              # outputs from preprocessing (created later)
```

---

## Notes & Tips

- **Determinism:** use `--seed` in preprocessing; set seeds in training if needed.  
- **Tokenizer padding:** training collator pads **labels** with `-100` to avoid loss on padding.  
- **Mixed precision:** `bf16` used if supported; otherwise `fp16` on CUDA.  
- **Adapters vs full model:** `infer_model.py` auto-detects adapter directories and requires `--base_model` in that case.

---

## Milestone 3 (Midterm) — What to Show

- **Problem & Audience:** why Steam price prediction matters, who uses it.  
- **Solution demo:** show preprocessing output, training command, and a couple of inference examples (single + batch with MAE).  
- **Infra:** Dockerized env; optional serverless (Modal/Cloud Run) later.  
- **Scalability:** QLoRA for memory efficiency; HF cache; autoscaling-ready containers.  
- **Future work:** alternative targets (tiers), additional signals (wishlists), model eval beyond MAE.

---

**License:** choose MIT/Apache-2.0 and add a `LICENSE` file.
