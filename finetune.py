#!/usr/bin/env python3
"""
Fine-tune a Qwen model to predict game prices as text (e.g., "12.99") using LoRA/optional QLoRA.

Dataset format (JSONL):
- input: string with game metadata & review stats
- output: string target price with two decimals (e.g., "12.99")

Examples:
    # 1.5B, full precision/bf16 (recommended baseline)
    python finetune.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --train_jsonl steam_out/qwen_train.jsonl \
        --val_jsonl   steam_out/qwen_val.jsonl \
        --output_dir  qwen-price-sft-1p5b \
        --epochs 3 --lr 2e-4 --batch_size 4 --grad_accum 4 --max_len 1024

    # Optional QLoRA (requires working bitsandbytes + CUDA):
    python finetune.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --train_jsonl steam_out/qwen_train.jsonl \
        --val_jsonl   steam_out/qwen_val.jsonl \
        --output_dir  qwen-price-sft-1p5b-qlora \
        --epochs 3 --lr 2e-4 --batch_size 4 --grad_accum 4 --max_len 1024 \
        --use_qlora
"""
import argparse, json, os, sys, re
from typing import Dict, List, Optional

# Stability on diverse GPU stacks (esp. H100) — disable exotic kernels by default.
os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLE", "1")
os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

# Optional PEFT / bitsandbytes
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    peft_available = True
except Exception:
    peft_available = False

try:
    import bitsandbytes as bnb  # noqa: F401
    bnb_available = True
except Exception:
    bnb_available = False

SYSTEM_PROMPT = (
    "You are a pricing model specialized in predicting Steam game prices in USD. "
    "Given game metadata and review stats, answer with ONLY the USD price using two decimals, e.g., 12.99"
)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

class PriceDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 1024):
        self.items = list(read_jsonl(path))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        inp = ex["input"]
        out = str(ex["output"]).strip()  # e.g., "12.99"

        prompt = f"{SYSTEM_PROMPT}\n\nGame metadata:\n{inp}\n\nPrice in USD:"
        # Only train on the answer; mask the prompt with -100 in labels
        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        answer_ids = self.tok(out + self.tok.eos_token, add_special_tokens=False).input_ids

        input_ids = (prompt_ids + answer_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_len]

        # attention mask for the visible tokens
        attn = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

class RightPadCollator:
    """
    Pads input_ids, attention_mask, and labels to the longest length in the batch.
    Labels are padded with -100 so the loss ignores padding.
    """
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        import torch
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            att = f["attention_mask"]
            lab = f["labels"]
            pad = max_len - ids.size(0)
            if pad > 0:
                ids = torch.nn.functional.pad(ids, (0, pad), value=self.pad_id)
                att = torch.nn.functional.pad(att, (0, pad), value=0)
                lab = torch.nn.functional.pad(lab, (0, pad), value=-100)
            input_ids.append(ids)
            attention_mask.append(att)
            labels.append(lab)
        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }

def get_model_and_tokenizer(model_name: str, use_qlora: bool):
    # Choose dtype safely (bf16 on H100/modern GPUs)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # QLoRA only if CUDA + bitsandbytes available
    load_kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=True)
    if use_qlora:
        if not torch.cuda.is_available() or not bnb_available:
            print("[WARN] --use_qlora requested but CUDA/bitsandbytes not available; using full precision.", flush=True)
            use_qlora = False
        else:
            load_kwargs.update(
                dict(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                )
            )

    # Load model (retry fallback if quantized load fails)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as e:
        print("[WARN] First load attempt failed:", e)
        print("[INFO] Retrying without 4-bit quantization.")
        load_kwargs.pop("load_in_4bit", None)
        load_kwargs.pop("bnb_4bit_compute_dtype", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        use_qlora = False  # we fell back

    # If we actually did 4-bit, prepare LoRA
    if use_qlora and peft_available:
        model = prepare_model_for_kbit_training(model)
        lconf = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lconf)

    return model, tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--use_qlora", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, tok = get_model_and_tokenizer(args.model, args.use_qlora)

    train_ds = PriceDataset(args.train_jsonl, tok, max_len=args.max_len)
    val_ds   = PriceDataset(args.val_jsonl,   tok, max_len=args.max_len)

    # Keep metrics stubbed (you can add MAE later)
    def compute_metrics(_):
        return {}

    # Use a version-safe TrainingArguments block (no evaluation_strategy for older TF)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
        bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        save_total_limit=2,
        logging_steps=50,  # ignored if your transformers is too old; harmless
    )

    # Critical fix: pad inputs **and labels** consistently
    data_collator = RightPadCollator(tok)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tok,  # FutureWarning is harmless; ok for TF<5.0
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Training complete. Model saved to", args.output_dir)

if __name__ == "__main__":
    main()
