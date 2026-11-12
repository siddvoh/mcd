import argparse
import base64
import csv
import hashlib
import io
import json
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
URL = "https://openrouter.ai/api/v1/chat/completions"
ITEMS = {"chestxray14": "chest64", "slake": "slake50", "bbbc039": "bbbc200"}
ANSWERS = {"chestxray14": ["no", "unknown", "yes"], "slake": ["heart", "lung", "no", "unknown", "yes"]}
COLUMNS = ["model", "cell", "item_id", "image_id", "colormap", "repeat", "answer", "abstain", "confidence", "reasoning_tokens"]
NO_RESPONSE_FORMAT = {"qwen3_vl_235b_or"}  # its provider rejects response_format, so only the prompt asks for JSON
MAX_TRIES = 10


def load_items(dataset):
    return pd.read_csv(ROOT / "data" / "items" / f"{ITEMS[dataset]}.csv", dtype=str, keep_default_na=False)


def prompts(plan, dataset, arm, question):
    user = plan["prompts"][dataset]["user"]
    if arm == "reminder":
        user = user.rstrip() + "\n" + plan["prompts"]["reminder"]
    return plan["prompts"][dataset]["system"], user.replace("{question}", question)


def load_scalar(path, cap):
    pil = Image.open(path)
    if pil.mode in ("P", "I;16", "I"):
        pil = pil.convert("L")
    elif pil.mode == "RGBA":
        pil = pil.convert("RGB")
    arr = np.array(pil)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    arr = arr.astype(np.float32)
    h, w = arr.shape
    if max(h, w) > cap:
        scale = cap / float(max(h, w))
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        arr = np.asarray(Image.fromarray(arr).resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    return arr


def render_png(scalar, cmap):
    arr = scalar.astype(np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    normed = np.zeros_like(arr) if hi <= lo else np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    rgb = matplotlib.colormaps[cmap](normed)[..., :3].astype(np.float32)
    buf = io.BytesIO()
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def schedule(item_ids, maps, repeats, seed):
    order = []
    for index, item_id in enumerate(item_ids):
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{item_id}".encode()).digest()[:8], "big"))
        base = [maps[k] for k in rng.permutation(len(maps))]
        calls = []
        for block, repeat in enumerate(int(r) for r in rng.permutation(repeats)):
            shift = (index + block) % len(base)
            row = base[shift:] + base[:shift]
            if ((index // len(base)) + block) % 2:
                row = row[::-1]
            calls.extend((cmap, repeat) for cmap in row)
        order.extend((item_id, cmap, repeat) for cmap, repeat in calls)
    return order


def response_format(dataset):
    if dataset == "bbbc039":
        name, extra, required = "count_response", {"task": {"type": "string", "enum": ["count"]}, "count": {"type": "integer"}}, ["task", "count"]
    else:
        name = "slake_medical_qa_response" if dataset == "slake" else "medical_qa_response"
        extra = {"task": {"type": "string", "enum": ["medical_qa"]}, "answer": {"type": "string", "enum": ANSWERS[dataset]}}
        required = ["task", "answer"]
    properties = {**extra, "abstain": {"type": "boolean"}, "confidence": {"type": "number"}}
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": required + ["abstain", "confidence"], "properties": properties}}}


def parse(text, dataset):
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if dataset == "bbbc039":
        count = obj.get("count")
        if set(obj) != {"task", "count", "abstain", "confidence"} or obj.get("task") != "count":
            return None
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
    else:
        answer = obj.get("answer")
        if set(obj) != {"task", "answer", "abstain", "confidence"} or obj.get("task") != "medical_qa":
            return None
        if not isinstance(answer, str) or answer.strip().lower() not in ANSWERS[dataset]:
            return None
    confidence = obj.get("confidence")
    if not isinstance(obj.get("abstain"), bool) or isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    return obj if 0.0 <= float(confidence) <= 1.0 else None


def request_body(plan, model, dataset, system, user, png):
    settings = plan["settings"]
    body = {
        "model": plan["models"][model]["id"],
        "provider": {"order": [plan["models"][model]["provider"]], "allow_fallbacks": False},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            ]},
        ],
        "max_tokens": settings["max_tokens"],
        "temperature": settings["temperature"],
        "include_reasoning": True,
        "reasoning": {"effort": settings["reasoning_effort"]},
    }
    if model not in NO_RESPONSE_FORMAT:
        body["response_format"] = response_format(dataset)
    return body


def ask(session, body, dataset):
    for attempt in range(MAX_TRIES):
        try:
            payload = session.post(URL, json=body, timeout=120).json()
        except (requests.RequestException, ValueError):
            payload = {}
        choice = (payload.get("choices") or [{}])[0]
        if payload.get("model") == body["model"] and choice.get("finish_reason") == "stop":
            answer = parse(choice["message"].get("content") or "", dataset)
            if answer is not None:
                return answer, payload.get("usage") or {}
        time.sleep(min(60, 2 ** attempt))
    raise SystemExit(f"no valid answer after {MAX_TRIES} tries")


def run_cell(plan, model, cell, out):
    spec = next(c for c in plan["cells"] if c["name"] == cell)
    dataset = spec["dataset"]
    items = load_items(dataset).set_index("item_id")
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["OPENROUTER_API_KEY"]
    scalars = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for item_id, cmap, repeat in schedule(list(items.index), spec["maps"], spec["repeats"], plan["schedule_seed"]):
            row = items.loc[item_id]
            if item_id not in scalars:
                scalars[item_id] = load_scalar(ROOT / row["image_path"], plan["settings"]["image_long_edge_px"])
            system, user = prompts(plan, dataset, spec["arm"], row.get("question", ""))
            body = request_body(plan, model, dataset, system, user, render_png(scalars[item_id], cmap))
            answer, usage = ask(session, body, dataset)
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            writer.writerow([model, cell, item_id, row["image_id"], cmap, repeat,
                             answer["count"] if dataset == "bbbc039" else answer["answer"],
                             str(answer["abstain"]).lower(), answer["confidence"], reasoning])
            handle.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    plan = json.loads((ROOT / "data" / "plan.json").read_text())
    run_cell(plan, args.model, args.cell, args.out or ROOT / "runs" / args.model / f"{args.cell}.csv")


if __name__ == "__main__":
    main()
