#!/usr/bin/env python3
# Predict Steam game price from the same prompt format used during training.

import argparse, json, os, re, sys
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional PEFT loading (for LoRA adapters)
try:
    from peft import PeftModel
    peft_available = True
except Exception:
    peft_available = False

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

def first_number(text):
    # extract a number like 12 or 12.99 (ignore thousands separators)
    m = re.search(r"(\d+(?:\.\d{1,2})?)", text.replace(",", ""))
    return float(m.group(1)) if m else None

def build_prompt(inp: str) -> str:
    # Must mirror training exactly
    return f"{SYSTEM_PROMPT}\n\nGame metadata:\n{inp}\n\nPrice in USD:"

def load_model_and_tokenizer(model_or_adapter: str, base_model: str = None):
    """
    - If `model_or_adapter` is a full model dir, load it directly.
    - If it contains a LoRA adapter (adapter_config.json) and you pass --base_model,
      load base then attach adapter.
    """
    # tokenizer always taken from the base (or the same dir for full model)
    tok_src = base_model or model_or_adapter
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # detect adapter directory
    is_adapter = os.path.exists(os.path.join(model_or_adapter, "adapter_config.json"))

    if is_adapter:
        if not peft_available:
            raise RuntimeError(
                "This path looks like a LoRA adapter but `peft` isn't installed. "
                "Install peft or provide a full model path."
            )
        if not base_model:
            raise RuntimeError(
                "LoRA adapter detected. Please provide --base_model (e.g., Qwen/Qwen2.5-1.5B-Instruct)."
            )
        base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, model_or_adapter)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_or_adapter, trust_remote_code=True)

    model.to(device).eval()
    return model, tok, device

def generate_price(model, tok, device, inp, max_new_tokens=16):
    prompt = build_prompt(inp)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    # take the suffix after the last "Price in USD:"
    suffix = text.split("Price in USD:")[-1].strip()
    val = first_number(suffix)
    return suffix, val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="Path to fine-tuned model directory OR LoRA adapter directory.")
    ap.add_argument("--base_model", default=None,
                    help="Base model name (only needed when --model_path is a LoRA adapter).")
    ap.add_argument("--input_jsonl", default=None,
                    help="JSONL with rows containing {'input': str, 'appid'?, 'price_usd'?}.")
    ap.add_argument("--text_input", default=None, help="Single raw text block with the game metadata.")
    ap.add_argument("--out", default=None, help="Where to save JSONL predictions for batch mode.")
    args = ap.parse_args()

    model, tok, device = load_model_and_tokenizer(args.model_path, args.base_model)

    # Single text mode
    if args.text_input:
        pred_text, pred_val = generate_price(model, tok, device, args.text_input)
        print("MODEL OUTPUT:", pred_text)
        print("PARSED PRICE:", pred_val)
        return

    # Batch mode
    if args.input_jsonl:
        outs = []
        mae_sum = 0.0
        mae_n = 0
        for r in read_jsonl(args.input_jsonl):
            inp = r["input"]
            appid = r.get("appid")
            gt = r.get("price_usd")
            if gt is not None:
                try:
                    gt = float(gt)
                except Exception:
                    gt = None
            pred_text, pred_val = generate_price(model, tok, device, inp)
            item = {
                "appid": appid,
                "prediction_text": pred_text,
                "prediction_value": pred_val,
                "gt_price_usd": gt
            }
            outs.append(item)
            if pred_val is not None and gt is not None:
                mae_sum += abs(pred_val - gt)
                mae_n += 1

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                for o in outs:
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
            print("Wrote predictions to", args.out)

        if mae_n > 0:
            print(f"MAE over {mae_n} items: {mae_sum/mae_n:.4f}")
        else:
            print("No MAE computed (need both parsed preds and ground truth).")
        return

    print("Nothing to do. Provide --text_input or --input_jsonl.")
    sys.exit(1)

if __name__ == "__main__":
    main()
