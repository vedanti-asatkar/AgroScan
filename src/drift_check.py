"""Data-drift check: training vs incoming images (class-balanced sample).

KS test on mean brightness and per-channel means; writes reports/drift.json.
Exits 1 on drift only if drift.fail_on_drift is true (default false).
"""
import random
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy import stats

from src.common import load_params, parse_args, read_manifest, write_json


def balanced_sample(rows, per_class, seed=0):
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r["path"])
    rng = random.Random(seed)
    out = []
    for paths in by.values():
        rng.shuffle(paths)
        out += paths[:per_class]
    return out


def features(paths):
    feats = []
    for p in paths:
        try:
            a = np.asarray(Image.open(p).convert("RGB").resize((64, 64)), dtype=np.float32)
        except Exception:
            continue
        feats.append([a.mean(), *a.reshape(-1, 3).mean(0)])
    return np.array(feats)


def main(params_path):
    p = load_params(params_path)
    prep = p["data"]["prepared_dir"]
    n = p["drift"]["max_per_class"]
    ref = features(balanced_sample(read_manifest(f"{prep}/train.csv"), n))
    cur = features(balanced_sample(read_manifest(f"{prep}/test.csv"), n))
    result, drift = {}, False
    for i, name in enumerate(["brightness", "red_mean", "green_mean", "blue_mean"]):
        ks, pv = stats.ks_2samp(ref[:, i], cur[:, i])
        flag = bool(pv < p["drift"]["alpha"])
        drift |= flag
        result[name] = {"ks_stat": float(ks), "p_value": float(pv), "drift": flag}
    result["n_reference"], result["n_incoming"], result["drift_detected"] = len(ref), len(cur), drift
    write_json(result, "reports/drift.json")
    print(result)
    if drift:
        print("DRIFT DETECTED")
        if p["drift"].get("fail_on_drift", False):
            sys.exit(1)


if __name__ == "__main__":
    main(parse_args("Drift check").params)
