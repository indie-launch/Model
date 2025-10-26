#!/usr/bin/env python3
"""
Preprocess Steam JSONL dumps into (a) a Qwen SFT dataset and (b) a numeric CSV for baseline models.

Inputs (JSONL):
- steam-app-list.jsonl                -> {"appid": int, "name": str}
- steam-app-reviews.jsonl             -> {"appid": int, "total_positive": int, "total_negative": int, "total_reviews": int, "num_reviews": int, "review_score": int, "review_score_desc": str}
- steam-raw-app-data.jsonl            -> Steam store app object including "steam_appid", "genres", "categories", "price_overview", "is_free", "short_description", etc.

Outputs:
- out_dir/qwen_train.jsonl            -> [{"input": "...", "output": "12.99", "appid": 12345, "price_usd": 12.99}, ...]
- out_dir/qwen_val.jsonl
- out_dir/features.csv                -> numeric features (for quick baselines)
- out_dir/id_map.csv                  -> (appid, name) mapping
- out_dir/readme.txt                  -> brief schema notes

Usage:
    python preprocess_data.py \
        --app-list /path/steam-app-list.jsonl \
        --reviews /path/steam-app-reviews.jsonl \
        --raw /path/steam-raw-app-data.jsonl \
        --out-dir ./data \
        --min-total-reviews 10 \
        --seed 42 \
        --val-ratio 0.1
"""
import argparse, json, os, math, random, csv, re
from datetime import datetime

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def parse_price_usd(app):
    # Prefer price_overview.final in USD; falls back to 0 for free games.
    is_free = bool(app.get("is_free", False))
    pov = app.get("price_overview")
    if pov and pov.get("currency") == "USD":
        final_cents = pov.get("final")
        if isinstance(final_cents, (int, float)):
            return round(float(final_cents) / 100.0, 2)
    if is_free:
        return 0.0
    # Could not resolve a USD price
    return None

def parse_release_year(app):
    date_obj = app.get("release_date", {})
    date_str = date_obj.get("date")
    if not date_str:
        return None
    # Try to extract a 4-digit year
    m = re.search(r"(19|20)\d{2}", date_str)
    if m:
        return int(m.group(0))
    return None

def join_appid_maps(app_list_path):
    id2name = {}
    for row in read_jsonl(app_list_path):
        aid = row.get("appid")
        nm  = row.get("name")
        if isinstance(aid, int):
            id2name[aid] = nm
    return id2name

def join_reviews_map(reviews_path):
    # Keep last occurrence if dupes
    rmap = {}
    for row in read_jsonl(reviews_path):
        aid = row.get("appid")
        if not isinstance(aid, int):
            continue
        rmap[aid] = {
            "num_reviews": row.get("num_reviews"),
            "review_score": row.get("review_score"),
            "review_score_desc": row.get("review_score_desc"),
            "total_positive": row.get("total_positive"),
            "total_negative": row.get("total_negative"),
            "total_reviews": row.get("total_reviews"),
        }
    return rmap

def build_text_input(app, reviews_row):
    name = app.get("name") or ""
    genres = ", ".join([g.get("description","") for g in app.get("genres", []) if isinstance(g, dict)])
    categories = ", ".join([c.get("description","") for c in app.get("categories", []) if isinstance(c, dict)])
    short_desc = (app.get("short_description") or "").strip()
    short_desc = re.sub(r"\s+", " ", short_desc)
    devs = ", ".join(app.get("developers", [])) if isinstance(app.get("developers"), list) else (app.get("developers") or "")
    pubs = ", ".join(app.get("publishers", [])) if isinstance(app.get("publishers"), list) else (app.get("publishers") or "")
    is_free = bool(app.get("is_free", False))
    release_year = parse_release_year(app)
    plat = app.get("platforms") or {}
    platforms = ", ".join([k for k,v in plat.items() if v]) if isinstance(plat, dict) else ""

    if reviews_row:
        rs_desc = reviews_row.get("review_score_desc")
        tot_pos = reviews_row.get("total_positive")
        tot_neg = reviews_row.get("total_negative")
        tot_all = reviews_row.get("total_reviews")
        pos_ratio = (float(tot_pos) / float(tot_all)) if (isinstance(tot_pos, (int,float)) and isinstance(tot_all, (int,float)) and tot_all>0) else None
        review_block = f"Review Summary: {rs_desc}; Positive: {tot_pos}; Negative: {tot_neg}; Total: {tot_all}; Positive_Ratio: {pos_ratio:.4f}" if pos_ratio is not None else f"Review Summary: {rs_desc}; Positive: {tot_pos}; Negative: {tot_neg}; Total: {tot_all}"
    else:
        review_block = "Review Summary: N/A"

    txt = (
        f"Name: {name}\n"
        f"Genres: {genres}\n"
        f"Categories: {categories}\n"
        f"Developers: {devs}\n"
        f"Publishers: {pubs}\n"
        f"Platforms: {platforms}\n"
        f"Is Free: {is_free}\n"
        f"Release Year: {release_year}\n"
        f"{review_block}\n"
        f"Short Description: {short_desc}\n"
    )
    return txt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-list", required=True, help="Path to steam-app-list.jsonl")
    ap.add_argument("--reviews", required=True, help="Path to steam-app-reviews.jsonl")
    ap.add_argument("--raw", required=True, help="Path to steam-raw-app-data.jsonl")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--min-total-reviews", type=int, default=0, help="Filter out games with fewer than this many total reviews")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    id2name = join_appid_maps(args.app_list)
    rmap = join_reviews_map(args.reviews)

    rows = []
    id_map_rows = []

    kept = 0
    skipped_price = 0
    skipped_reviews = 0

    for app in read_jsonl(args.raw):
        appid = app.get("steam_appid")
        if not isinstance(appid, int):
            continue

        price_usd = parse_price_usd(app)
        if price_usd is None:
            skipped_price += 1
            continue

        rv = rmap.get(appid)
        if rv:
            tot_all = rv.get("total_reviews")
            if isinstance(tot_all, int) and tot_all < args.min_total_reviews:
                skipped_reviews += 1
                continue

        text_input = build_text_input(app, rv)
        item = {
            "appid": appid,
            "name": app.get("name") or id2name.get(appid),
            "input": text_input,
            "output": f"{price_usd:.2f}",
            "price_usd": price_usd,
            "is_free": bool(app.get("is_free", False)),
            "review_score_desc": rv.get("review_score_desc") if rv else None,
            "total_positive": rv.get("total_positive") if rv else None,
            "total_negative": rv.get("total_negative") if rv else None,
            "total_reviews": rv.get("total_reviews") if rv else None,
            "num_reviews": rv.get("num_reviews") if rv else None,
            "genres": [g.get("description","") for g in app.get("genres", []) if isinstance(g, dict)],
            "categories": [c.get("description","") for c in app.get("categories", []) if isinstance(c, dict)],
            "release_year": parse_release_year(app),
        }
        rows.append(item)
        id_map_rows.append({"appid": appid, "name": item["name"]})
        kept += 1

    random.seed(args.seed)
    random.shuffle(rows)
    n = len(rows)
    n_val = max(1, int(n * args.val_ratio)) if n>0 else 0

    val = rows[:n_val]
    train = rows[n_val:]

    # Save Qwen SFT jsonl
    with open(os.path.join(args.out_dir, "qwen_train.jsonl"), "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps({"input": r["input"], "output": r["output"], "appid": r["appid"], "price_usd": r["price_usd"]}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "qwen_val.jsonl"), "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps({"input": r["input"], "output": r["output"], "appid": r["appid"], "price_usd": r["price_usd"]}, ensure_ascii=False) + "\n")

    # Save numeric features CSV
    # Columns: appid, price_usd, is_free, total_positive, total_negative, total_reviews, pos_ratio, num_reviews, release_year
    import csv
    feat_path = os.path.join(args.out_dir, "features.csv")
    with open(feat_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["appid","price_usd","is_free","total_positive","total_negative","total_reviews","pos_ratio","num_reviews","release_year"])
        for r in rows:
            tp = r["total_positive"] or 0
            tn = r["total_negative"] or 0
            tr = r["total_reviews"] or (tp+tn)
            pos_ratio = (float(tp)/float(tr)) if tr else 0.0
            writer.writerow([r["appid"], r["price_usd"], int(r["is_free"]), tp, tn, tr, f"{pos_ratio:.6f}", r["num_reviews"] or tr, r["release_year"] or ""])

    # Save id map
    with open(os.path.join(args.out_dir, "id_map.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["appid","name"])
        for m in id_map_rows:
            writer.writerow([m["appid"], m["name"] or ""])

    # readme
    with open(os.path.join(args.out_dir, "readme.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Files:\n"
            "- qwen_train.jsonl / qwen_val.jsonl: supervised fine-tune pairs for Qwen (fields: input, output (price string), appid, price_usd)\n"
            "- features.csv: numeric features; you can try a quick baseline (e.g., XGBoost) if desired\n"
            "- id_map.csv: appid->name\n\n"
            f"Kept: {kept} | Skipped (no price): {skipped_price} | Skipped (few reviews): {skipped_reviews}\n"
        )

    print(f"Done. Train: {len(train)}, Val: {len(val)}, Total: {len(rows)}")
    print(f"Wrote to: {args.out_dir}")

if __name__ == "__main__":
    main()
