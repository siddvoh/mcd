import argparse
import hashlib
import json
import math
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

import lut_validation

ROOT = Path(__file__).resolve().parent
MODELS = {
    "claude_sonnet_4_6_or": ("Claude", "Claude Sonnet 4.6"),
    "gemini_3_1_pro_or": ("Gemini", "Gemini 3.1 Pro"),
    "gpt_5_4_or": ("GPT", "GPT-5.4"),
    "qwen3_vl_235b_or": ("Qwen", "Qwen3-VL-235B"),
}
PANELS = {"bbbc039": "BBBC039", "chestxray14": "ChestX-ray14", "slake": "SLAKE"}
MAPS = ("gray", "viridis", "plasma", "inferno", "magma", "cividis")
RAW = {"chestxray14": "chest64_multirepeat", "slake": "slake50_raw", "bbbc039": "bbbc200_raw"}
SAME = {"chestxray14": "chest64_same_gray", "slake": "slake50_same_gray", "bbbc039": "bbbc200_same_gray"}
REMINDER = {"chestxray14": "chest64_reminder", "slake": "slake50_reminder"}
N_BOOT = 10_000
N_PERM = 5_000
N_FLIP_NULL = 2_000
FIGURE_IMAGE = "chestxray14_atelectasis_yes_021"


def entropy(labels):
    counts = np.asarray(sorted(Counter(labels).values()), dtype=float)
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def plugin_mi(groups):
    pooled = [a for g in groups for a in g]
    return entropy(pooled) - float(np.mean([entropy(g) for g in groups]))


def seed_of(label):
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")


def mean_ci(values, label, clusters=None):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed_of(label))
    if clusters is None:
        draws = values[rng.integers(0, values.size, size=(N_BOOT, values.size))].mean(axis=1)
    else:
        clusters = np.asarray(clusters, dtype=str)
        unique = np.unique(clusters)
        draws = np.empty(N_BOOT)
        for i in range(N_BOOT):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            draws[i] = np.concatenate([values[clusters == c] for c in chosen]).mean()
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(items, metrics, keys):
    rows = []
    for key, group in items.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        row["n_items"] = len(group)
        row["n_images"] = int(group["image_id"].nunique())
        clusters = group["image_id"] if row.get("dataset") == "slake" else None
        for metric in metrics:
            label = "|".join(map(str, key)) + "|" + metric
            row[metric], row[metric + "_lo"], row[metric + "_hi"] = mean_ci(group[metric], label, clusters)
        rows.append(row)
    return pd.DataFrame(rows)


def load_items():
    items = {}
    for name in ("chest64", "slake50", "bbbc200"):
        frame = pd.read_csv(ROOT / "data" / "items" / f"{name}.csv", dtype=str, keep_default_na=False)
        truth = frame["ground_truth_count"] if name == "bbbc200" else frame["answer"]
        frame["truth"] = truth.str.strip().str.lower()
        items[name] = frame
    return items


def load_answers(plan, items):
    data = pd.read_csv(ROOT / "data" / "answers.csv", dtype=str, keep_default_na=False)
    data = data.rename(columns={"colormap": "cmap", "repeat": "seed"})
    cells = {spec["name"]: spec for spec in plan["cells"]}
    data["dataset"] = data["cell"].map(lambda c: cells[c]["dataset"])
    data["seed"] = data["seed"].astype(int)
    data["confidence"] = data["confidence"].astype(float)
    data["reasoning_tokens"] = data["reasoning_tokens"].astype(int)
    data["answer"] = np.where(data["abstain"] == "true", "__abstain__", data["answer"].str.strip().str.lower())
    truth = {}
    for spec in plan["cells"]:
        frame = items[spec["items"].removesuffix(".csv")]
        truth.update(zip(frame["item_id"], frame["truth"]))
        want = {(i, m, r) for i in frame["item_id"] for m in spec["maps"] for r in range(spec["repeats"])}
        for model in MODELS:
            part = data[(data["cell"] == spec["name"]) & (data["model"] == model)]
            if len(part) != len(want) or set(zip(part["item_id"], part["cmap"], part["seed"])) != want:
                raise SystemExit(f"{model}/{spec['name']}: answers do not match the plan")
    data["truth"] = data["item_id"].map(truth)
    return data


def orbit(item, seed=0, drop=None):
    frame = item[item["seed"] == seed]
    if drop is not None:
        frame = frame[frame["cmap"] != drop]
    return entropy(frame["answer"])


def repeated_decode(data):
    source = data[data["cell"] == "chest64_multirepeat"]
    rows = []
    for (model, item_id), item in source.groupby(["model", "item_id"]):
        groups = [g.sort_values("seed")["answer"].tolist() for _, g in item.groupby("cmap")]
        rows.append({
            "model": model, "item_id": item_id, "image_id": item["image_id"].iloc[0],
            "r0": orbit(item, 0), "r1": orbit(item, 1), "r2": orbit(item, 2),
            "mi": plugin_mi(groups),
            "within": float(np.mean([entropy(g) for g in groups])),
        })
    items = pd.DataFrame(rows)
    # summarize() seeds each bootstrap from the column name, so these names fix the published intervals.
    names = {"r0": "one_repeat_score_r0", "r1": "one_repeat_score_r1", "r2": "one_repeat_score_r2",
             "mi": "three_repeat_plugin_mi", "within": "conditional_noise"}
    summary = summarize(items.rename(columns=names), list(names.values()), ["model"])
    return summary.rename(columns={v + s: k + s for k, v in names.items() for s in ("", "_lo", "_hi")})


def permutation_test(data):
    source = data[data["cell"] == "chest64_multirepeat"]
    rows = []
    for model, frame in source.groupby("model"):
        observed, blocks_by_item = [], []
        for _, item in frame.groupby("item_id"):
            cmaps = sorted(item["cmap"].unique())
            groups = [item[item["cmap"] == c].sort_values("seed")["answer"].tolist() for c in cmaps]
            blocks = [item[item["seed"] == s].set_index("cmap").loc[cmaps, "answer"].tolist()
                      for s in sorted(item["seed"].unique())]
            observed.append(max(0.0, min(1.0, plugin_mi(groups) / math.log(6))))
            blocks_by_item.append(blocks)
        rng = np.random.default_rng(seed_of(f"{model}:colormap-permutation"))
        null = np.empty(N_PERM)
        for k in range(N_PERM):
            scores = []
            for blocks in blocks_by_item:
                permuted = [[] for _ in blocks[0]]
                for block in blocks:
                    for index, answer in enumerate(rng.permutation(block)):
                        permuted[index].append(answer)
                scores.append(max(0.0, min(1.0, plugin_mi(permuted) / math.log(6))))
            null[k] = np.mean(scores)
        stat = float(np.mean(observed))
        rows.append({"model": model, "p": float((1 + np.sum(null >= stat)) / (N_PERM + 1)),
                     "null_mean": float(null.mean() * math.log(6))})
    frame = pd.DataFrame(rows).sort_values(["p", "model"]).reset_index(drop=True)
    running = 0.0
    adjusted = []
    for index, p in enumerate(frame["p"]):
        running = max(running, min(1.0, p * (len(frame) - index)))
        adjusted.append(running)
    frame["p_holm"] = adjusted
    return frame.sort_values("model").reset_index(drop=True)


def paired_six(data, dataset, cell):
    raw = data[data["cell"] == RAW[dataset]]
    raw = raw[raw["seed"] == 0]
    other = data[data["cell"] == cell]
    for model in sorted(raw["model"].unique()):
        a, b = raw[raw["model"] == model], other[other["model"] == model]
        for item_id, raw_item in a.groupby("item_id"):
            yield model, item_id, raw_item, b[b["item_id"] == item_id]


def matched(data):
    rows = []
    for dataset in RAW:
        for model, item_id, raw_item, gray in paired_six(data, dataset, SAME[dataset]):
            rows.append({
                "model": model, "dataset": dataset, "item_id": item_id, "image_id": raw_item["image_id"].iloc[0],
                "raw_orbit_score": orbit(raw_item), "same_gray_score": entropy(gray["answer"]),
            })
    items = pd.DataFrame(rows)
    items["raw_minus_same_gray"] = items["raw_orbit_score"] - items["same_gray_score"]
    return items, summarize(items, ["raw_orbit_score", "same_gray_score", "raw_minus_same_gray"], ["model", "dataset"])


def aurc(errors, uncertainty):
    errors = np.asarray(errors, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    area, kept_errors, kept = 0.0, 0.0, 0
    for value in np.unique(uncertainty):
        block = errors[uncertainty == value]
        kept_errors += block.sum()
        kept += len(block)
        area += kept_errors / kept * len(block) / len(errors)
    return float(area)


def selective(data):
    rows = []
    scorers = {"orbit": "orbit_u", "same_gray": "same_u", "confidence": "conf_u"}
    for dataset in RAW:
        by_model = {}
        for model, item_id, raw_item, gray in paired_six(data, dataset, SAME[dataset]):
            shared = raw_item[raw_item["cmap"] == "gray"].iloc[0]
            gray_by_seed = dict(zip(gray["seed"], gray["answer"]))
            record = {
                "item_id": item_id, "image_id": raw_item["image_id"].iloc[0],
                "error": int(shared["answer"] == "__abstain__" or shared["answer"] != shared["truth"]),
                "shared_abstain": int(shared["answer"] == "__abstain__"),
                "orbit_abstain": int((raw_item["answer"] == "__abstain__").sum()),
                "same_abstain": int((gray["answer"] == "__abstain__").sum()),
                "orbit_u": entropy(raw_item["answer"]), "conf_u": 1.0 - shared["confidence"],
            }
            for omit in range(6):
                record[f"same_u{omit}"] = entropy([shared["answer"]] + [gray_by_seed[s] for s in range(6) if s != omit])
            record["same_u"] = record["same_u5"]
            by_model.setdefault(model, []).append(record)
        for model, records in by_model.items():
            pairs = pd.DataFrame(records)
            errors = pairs["error"].to_numpy(float)
            point = {k: aurc(errors, pairs[c]) for k, c in scorers.items()}
            rng = np.random.default_rng(seed_of(f"{model}:{dataset}:aurc-cluster-bootstrap"))
            clusters = pairs["image_id"].astype(str).unique()
            index = {c: pairs[pairs["image_id"].astype(str) == c] for c in clusters}
            deltas = {k: [] for k in scorers if k != "orbit"}
            for _ in range(N_BOOT):
                sample = pd.concat([index[c] for c in rng.choice(clusters, size=len(clusters), replace=True)])
                e = sample["error"].to_numpy(float)
                values = {k: aurc(e, sample[c]) for k, c in scorers.items()}
                for k in deltas:
                    deltas[k].append(values[k] - values["orbit"])
            rotations = [aurc(errors, pairs[f"same_u{s}"]) for s in range(6)]
            row = {"model": model, "dataset": dataset, "n_items": len(pairs), "n_images": len(clusters)}
            for k in scorers:
                row[f"{k}_aurc"] = point[k]
            for k, values in deltas.items():
                row[f"{k}_minus_orbit"] = point[k] - point["orbit"]
                row[f"{k}_minus_orbit_lo"] = float(np.quantile(values, 0.025))
                row[f"{k}_minus_orbit_hi"] = float(np.quantile(values, 0.975))
            row["same_gray_rotation_min"] = min(rotations)
            row["same_gray_rotation_max"] = max(rotations)
            row["rotations_preserve_sign"] = all(np.sign(r - point["orbit"]) == np.sign(point["same_gray"] - point["orbit"]) for r in rotations)
            row["shared_abstention_rate"] = pairs["shared_abstain"].mean()
            row["orbit_abstention_rate"] = pairs["orbit_abstain"].sum() / (6 * len(pairs))
            row["same_gray_abstention_rate"] = pairs["same_abstain"].sum() / (6 * len(pairs))
            rows.append(row)
    return pd.DataFrame(rows)


def gray_leverage(data):
    rows = []
    for dataset, cell in RAW.items():
        source = data[(data["cell"] == cell) & (data["seed"] == 0)]
        for (model, item_id), item in source.groupby(["model", "item_id"]):
            omitted = {m: orbit(item, drop=m) for m in MAPS}
            chromatic = float(np.mean([v for m, v in omitted.items() if m != "gray"]))
            rows.append({"model": model, "dataset": dataset, "item_id": item_id, "image_id": item["image_id"].iloc[0],
                         "without_gray": omitted["gray"], "mean_chromatic_omission": chromatic,
                         "gray_leverage": chromatic - omitted["gray"]})
    items = pd.DataFrame(rows)
    return summarize(items, ["without_gray", "mean_chromatic_omission", "gray_leverage"], ["model", "dataset"])


def reminder(data):
    rows = []
    for dataset, cell in REMINDER.items():
        for model, item_id, raw_item, told in paired_six(data, dataset, cell):
            a, b = raw_item.set_index("cmap").loc[list(MAPS)], told.set_index("cmap").loc[list(MAPS)]
            rows.append({
                "model": model, "dataset": dataset, "item_id": item_id, "image_id": raw_item["image_id"].iloc[0],
                "raw_score": orbit(raw_item), "reminder_score": orbit(told),
                "raw_accuracy": float((a["answer"] == a["truth"]).mean()),
                "reminder_accuracy": float((b["answer"] == b["truth"]).mean()),
                "raw_abstentions": int((a["answer"] == "__abstain__").sum()),
                "reminder_abstentions": int((b["answer"] == "__abstain__").sum()),
            })
    items = pd.DataFrame(rows)
    items["score_reduction"] = items["raw_score"] - items["reminder_score"]
    items["accuracy_change"] = items["reminder_accuracy"] - items["raw_accuracy"]
    summary = summarize(items, ["raw_score", "reminder_score", "score_reduction", "accuracy_change"], ["model", "dataset"])
    counts = items.groupby(["model", "dataset"], as_index=False)[["raw_abstentions", "reminder_abstentions"]].sum()
    summary = summary.merge(counts, on=["model", "dataset"], validate="one_to_one")
    for arm in ("raw", "reminder"):
        summary[f"{arm}_abstention_rate"] = summary[f"{arm}_abstentions"] / (len(MAPS) * summary["n_items"])
    return summary


def majority(answers):
    ranked = Counter(answers).most_common()
    return ranked[0][0] if len(ranked) == 1 or ranked[0][1] > ranked[1][1] else None


def per_map(data):
    rows, changes, flips, within, spread = [], [], [], [], []
    chest = data[data["cell"] == RAW["chestxray14"]]
    slake = data[(data["cell"] == RAW["slake"]) & (data["seed"] == 0)]
    same = data[data["cell"] == SAME["chestxray14"]]
    for dataset, frame in (("chestxray14", chest), ("slake", slake)):
        for (model, cmap), part in frame.groupby(["model", "cmap"]):
            rows.append({"model": model, "dataset": dataset, "cmap": cmap,
                         "accuracy": float((part["answer"] == part["truth"]).mean()),
                         "yes_rate": float((part["answer"] == "yes").mean()),
                         "abstention_rate": float((part["answer"] == "__abstain__").mean())})
        for (model, item_id), item in frame.groupby(["model", "item_id"]):
            gray, colored = item[item["cmap"] == "gray"], item[item["cmap"] != "gray"]
            change = {"model": model, "dataset": dataset, "item_id": item_id, "image_id": item["image_id"].iloc[0]}
            for name, test in (("accuracy", lambda a: a["answer"] == a["truth"]), ("yes_rate", lambda a: a["answer"] == "yes"),
                               ("abstention_rate", lambda a: a["answer"] == "__abstain__")):
                change[f"{name}_change"] = float(test(colored).mean() - test(gray).mean())
            changes.append(change)
    for (model, item_id), item in chest.groupby(["model", "item_id"]):
        by_map = {m: item[item["cmap"] == m].sort_values("seed")["answer"].tolist() for m in MAPS}
        top = {m: majority(a) for m, a in by_map.items()}
        flipped = [m for m in MAPS[1:] if top[m] is not None and top["gray"] is not None and top[m] != top["gray"]]
        gray_pool = by_map["gray"] + same[(same["model"] == model) & (same["item_id"] == item_id)].sort_values("seed")["answer"].tolist()
        flips.append({"model": model, "item_id": item_id, "flipped_maps": len(flipped), "gray_pool": gray_pool,
                      "to_abstain": sum(top[m] == "__abstain__" for m in flipped),
                      "gray_correct": top["gray"] == item["truth"].iloc[0]})
        gray_entropy = entropy(by_map["gray"])
        chromatic_entropy = float(np.mean([entropy(by_map[m]) for m in MAPS[1:]]))
        within.append({"model": model, "dataset": "chestxray14", "item_id": item_id, "image_id": item["image_id"].iloc[0],
                       "gray_within": gray_entropy, "chromatic_within": chromatic_entropy,
                       "chromatic_minus_gray_within": chromatic_entropy - gray_entropy})
    flips = pd.DataFrame(flips)
    flip_summary = []
    for model, part in flips.groupby("model"):
        correct = part[part["gray_correct"]]
        rng = np.random.default_rng(seed_of(f"{model}:majority-flip-null"))
        null_all, null_correct = np.zeros(N_FLIP_NULL), np.zeros(N_FLIP_NULL)
        for pool, is_correct in zip(part["gray_pool"], part["gray_correct"]):
            pool = np.asarray(pool, dtype=object)
            for b in range(N_FLIP_NULL):
                draws = [majority(list(pool[rng.integers(0, len(pool), 3)])) for _ in MAPS]
                hit = draws[0] is not None and any(d is not None and d != draws[0] for d in draws[1:])
                null_all[b] += hit
                null_correct[b] += hit and is_correct
        flip_summary.append({"model": model, "items": len(part), "items_flipped": int((part["flipped_maps"] > 0).sum()),
                             "map_flips": int(part["flipped_maps"].sum()), "map_flips_to_abstain": int(part["to_abstain"].sum()),
                             "map_pairs": len(part) * (len(MAPS) - 1),
                             "gray_correct_items": len(correct),
                             "gray_correct_items_flipped": int((correct["flipped_maps"] > 0).sum()),
                             "null_items_flipped": float(null_all.mean()),
                             "null_items_flipped_lo": float(np.quantile(null_all, 0.025)),
                             "null_items_flipped_hi": float(np.quantile(null_all, 0.975)),
                             "null_gray_correct_flipped": float(null_correct.mean()),
                             "null_gray_correct_flipped_lo": float(np.quantile(null_correct, 0.025)),
                             "null_gray_correct_flipped_hi": float(np.quantile(null_correct, 0.975))})
    for model, item_id, raw_item, gray in paired_six(data, "bbbc039", SAME["bbbc039"]):
        counts = [pd.to_numeric(a["answer"], errors="coerce").dropna().to_numpy() for a in (raw_item, gray)]
        if min(len(c) for c in counts) < 2:
            continue
        orbit_spread, same_spread = (float(np.mean(np.abs(c - np.median(c)))) for c in counts)
        spread.append({"model": model, "dataset": "bbbc039", "item_id": item_id, "image_id": raw_item["image_id"].iloc[0],
                       "orbit_count_spread": orbit_spread, "same_count_spread": same_spread,
                       "count_spread_excess": orbit_spread - same_spread})
    metrics = ["accuracy_change", "yes_rate_change", "abstention_rate_change"]
    changes = summarize(pd.DataFrame(changes), metrics, ["model", "dataset"])
    within = summarize(pd.DataFrame(within), ["gray_within", "chromatic_within", "chromatic_minus_gray_within"], ["model", "dataset"])
    spread = summarize(pd.DataFrame(spread), ["orbit_count_spread", "same_count_spread", "count_spread_excess"], ["model", "dataset"])
    return pd.DataFrame(rows), changes, pd.DataFrame(flip_summary), within, spread

def num(value, places=3):
    text = format(Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP), f".{places}f")
    if text.startswith("-") and Decimal(text) == 0:
        text = text[1:]
    return "$-$" + text[1:] if text.startswith("-") else text


def interval(row, stem, bold=True):
    lo, hi = row[stem + "_lo"], row[stem + "_hi"]
    text = f"{num(row[stem])} [{num(lo)}, {num(hi)}]"
    return rf"\textbf{{{text}}}" if bold and (lo > 0 or hi < 0) else text


def table_primary(matched_summary, selective_summary):
    lines = [r"\begin{tabular}{llrrrrrrl}", r"\toprule", r"Model & Panel & rows/images &",
             r"$\widehat D_{\mathrm{orbit}}$ &", r"$\widehat D_{\mathrm{same}}$ &", r"$\Delta\widehat D$ [95\% CI] &",
             r"orbit AURC &", r"same AURC &", r"$\Delta$AURC [95\% CI] \\", r"\midrule"]
    m = matched_summary.set_index(["model", "dataset"])
    s = selective_summary.set_index(["model", "dataset"])
    for index, (model, (short, _)) in enumerate(MODELS.items()):
        for dataset, panel in PANELS.items():
            a, b = m.loc[(model, dataset)], s.loc[(model, dataset)]
            lines.append(f"{short} & {panel} & {int(a['n_items'])}/{int(a['n_images'])} & {num(a['raw_orbit_score'])} & {num(a['same_gray_score'])} &")
            tail = f"{interval(a, 'raw_minus_same_gray')} & {num(b['orbit_aurc'])} & {num(b['same_gray_aurc'])}"
            delta = interval(b, "same_gray_minus_orbit")
            both = "textbf" in interval(a, "raw_minus_same_gray") and "textbf" in delta
            lines += [tail + " &", delta + r" \\"] if both else [tail + " & " + delta + r" \\"]
        if index < len(MODELS) - 1:
            lines.append(r"\addlinespace")
    return lines + [r"\bottomrule", r"\end{tabular}"]


def table_repeated(summary, perm):
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Model &", r"\shortstack{r0 drift\\\relax [95\% CI]} &",
             r"\shortstack{plug-in MI\\\relax [95\% CI]} &", r"\shortstack{within-map entropy\\\relax [95\% CI]} &",
             r"Holm $p$ \\", r"\midrule"]
    s, p = summary.set_index("model"), perm.set_index("model")
    for model, (short, _) in MODELS.items():
        row = s.loc[model]
        lines.append(f"{short} & {interval(row, 'r0', False)} & {interval(row, 'mi', False)} & "
                     f"{interval(row, 'within', False)} & {num(p.loc[model, 'p_holm'], 4)} \\\\")
    return lines + [r"\bottomrule", r"\end{tabular}"]


def table_gray(summary):
    lines = [r"\begin{tabular}{llrrl}", r"\toprule", r"Model & Panel &", r"$\widehat D_{-g}$ &",
             r"$\overline{\widehat D}_{-\mathrm{chrom}}$ &", r"gray leverage [95\% CI] \\", r"\midrule"]
    s = summary.set_index(["model", "dataset"])
    for model, (short, _) in MODELS.items():
        for dataset, panel in PANELS.items():
            row = s.loc[(model, dataset)]
            lines.append(f"{short} & {panel} & {num(row['without_gray'])} & {num(row['mean_chromatic_omission'])} & "
                         f"{interval(row, 'gray_leverage')} \\\\")
    return lines + [r"\bottomrule", r"\end{tabular}"]


def table_reminder(summary):
    lines = [r"\begin{tabular}{llrrll}", r"\toprule", r"Model & Panel &", r"raw drift &", r"reminder drift &",
             r"\shortstack{drift reduction\\\relax [95\% CI]} &", r"\shortstack{accuracy change\\\relax [95\% CI]} \\", r"\midrule"]
    s = summary.set_index(["model", "dataset"])
    for model, (short, _) in MODELS.items():
        for dataset in REMINDER:
            row = s.loc[(model, dataset)]
            lines.append(f"{short} & {PANELS[dataset]} & {num(row['raw_score'])} & {num(row['reminder_score'])} & "
                         f"{interval(row, 'score_reduction')} &")
            lines.append(interval(row, "accuracy_change") + r" \\")
    return lines + [r"\bottomrule", r"\end{tabular}"]


def table_per_map(rows):
    lines = [r"\begin{tabular}{l" + "r" * len(MAPS) + "}", r"\toprule", "Model & " + " & ".join(MAPS) + r" \\"]
    r = rows.set_index(["model", "dataset", "cmap"])
    for dataset, metric, title in (("chestxray14", "accuracy", "ChestX-ray14 accuracy"),
                                   ("chestxray14", "abstention_rate", "ChestX-ray14 abstention rate"),
                                   ("slake", "accuracy", "SLAKE accuracy")):
        lines += [r"\midrule", rf"\multicolumn{{{len(MAPS) + 1}}}{{l}}{{\emph{{{title}}}}} \\"]
        for model, (short, _) in MODELS.items():
            lines.append(short + " & " + " & ".join(num(r.loc[(model, dataset, m), metric], 2) for m in MAPS) + r" \\")
    return lines + [r"\bottomrule", r"\end{tabular}"]


def table_matrix(plan, items):
    labels = {"slake50_raw": "six-map raw", "slake50_same_gray": "six-repeat gray", "slake50_reminder": "reinforced reminder",
              "chest64_multirepeat": "repeated six-map", "chest64_same_gray": "six-repeat gray",
              "chest64_reminder": "reinforced reminder", "bbbc200_raw": "six-map raw", "bbbc200_same_gray": "six-repeat gray"}
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Panel & condition & rows & images & maps & calls per row \\", r"\midrule"]
    cells = {c["name"]: c for c in plan["cells"]}
    for name in labels:
        spec = cells[name]
        frame = items[spec["items"].removesuffix(".csv")]
        lines.append(f"{PANELS[spec['dataset']]} & {labels[name]} & {len(frame)} & {frame['image_id'].nunique()} & "
                     f"{len(spec['maps'])} & {len(spec['maps']) * spec['repeats']} \\\\")
    return lines + [r"\bottomrule", r"\end{tabular}"]


def panel_facts(items):
    slake = items["slake50"]
    organ = slake["answer"].str.lower().isin(["heart", "lung"]) | slake["question"].str.contains(r"\bor\b")
    bbbc = items["bbbc200"].drop_duplicates("image_id")
    clipped = bbbc["pixels_above_255"].astype(int) / (bbbc["height"].astype(int) * bbbc["width"].astype(int))
    return {
        "bbbc_clipped_images": int((clipped > 0).sum()),
        "bbbc_clipped_fraction_median": float(clipped.median()),
        "bbbc_clipped_fraction_max": float(clipped.max()),
        "slake_rows": len(slake), "slake_images": slake["image_id"].nunique(),
        "slake_yes_no": int((~organ).sum()), "slake_organ": int(organ.sum()),
        "slake_yes": int((slake.loc[~organ, "answer"].str.lower() == "yes").sum()),
        "slake_no": int((slake.loc[~organ, "answer"].str.lower() == "no").sum()),
        "chest_cases_per_finding_and_answer": sorted(set(items["chest64"].groupby(["finding", "answer"]).size().astype(int))),
        "chest_rows": len(items["chest64"]), "chest_images": items["chest64"]["image_id"].nunique(),
        "chest_patients": items["chest64"]["nih_image_index"].str.split("_").str[0].nunique(),
        "bbbc_rows": len(items["bbbc200"]), "bbbc_images": items["bbbc200"]["image_id"].nunique(),
    }


def reasoning_calls(data, above=0):
    return {model: int((part["reasoning_tokens"] > above).sum()) for model, part in data.groupby("model")}


def figure_facts(data, image_id=FIGURE_IMAGE):
    claude = data[(data["model"] == "claude_sonnet_4_6_or") & (data["dataset"] == "chestxray14")]
    raw, same = claude[claude["cell"] == RAW["chestxray14"]], claude[claude["cell"] == SAME["chestxray14"]]
    fixed = []
    for item_id, part in raw.groupby("item_id"):
        answers = part.groupby("cmap")["answer"].agg(set)
        if all(len(a) == 1 for a in answers) and len(set().union(*answers)) > 1:
            fixed.append(item_id)
    example = raw[raw["image_id"] == image_id].sort_values("seed")
    return {
        "claude_chest_fixed_per_map_split_across_maps": len(fixed),
        "of_which_six_gray_repeats_agree": sum(same.loc[same["item_id"] == i, "answer"].nunique() == 1 for i in fixed),
        "example_image": image_id,
        "example_answers": {m: list(example.loc[example["cmap"] == m, "answer"]) for m in MAPS},
        "example_gray_repeats": list(same.loc[same["image_id"] == image_id].sort_values("seed")["answer"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    out = parser.parse_args().out
    plan = json.loads((ROOT / "data" / "plan.json").read_text())
    items = load_items()
    data = load_answers(plan, items)
    rep = repeated_decode(data)
    perm = permutation_test(data)
    matched_items, matched_summary = matched(data)
    sel = selective(data)
    gray = gray_leverage(data)
    rem = reminder(data)
    lut_lines, lut = lut_validation.table()
    map_rows, map_changes, flips, within, spread = per_map(data)
    claude_slake = matched_items[(matched_items["model"] == "claude_sonnet_4_6_or") & (matched_items["dataset"] == "slake")]
    values = {
        "panels": panel_facts(items), "lut": lut,
        "claude_slake_same_gray_disagreeing_rows": int((claude_slake["same_gray_score"] > 0).sum()),
        "repeated_decode": rep.to_dict("records"), "permutation": perm.to_dict("records"),
        "matched": matched_summary.to_dict("records"), "selective": sel.to_dict("records"),
        "gray_leverage": gray.to_dict("records"), "reminder": rem.to_dict("records"),
        "per_map": map_rows.to_dict("records"), "per_map_change": map_changes.to_dict("records"),
        "majority_flips": flips.to_dict("records"),
        "within_map_by_display": within.to_dict("records"), "count_spread": spread.to_dict("records"),
        "reasoning_calls": reasoning_calls(data),
        "reasoning_calls_over_max_tokens": reasoning_calls(data, plan["settings"]["max_tokens"]),
        "figure": figure_facts(data),
    }
    tables = {
        "table_primary_results": table_primary(matched_summary, sel),
        "table_repeated_results": table_repeated(rep, perm),
        "table_gray_results": table_gray(gray),
        "table_reminder_results": table_reminder(rem),
        "table_colormap_validation": lut_lines,
        "table_validation_matrix": table_matrix(plan, items),
        "table_per_map": table_per_map(map_rows),
    }
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for name, lines in tables.items():
        (out / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n")
    (out / "values.json").write_text(json.dumps(values, indent=1, sort_keys=True, default=lambda v: v.item()) + "\n")


if __name__ == "__main__":
    main()
