
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def metrics_from_confusion(cm) -> dict:
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("a confusion matrix must be square")
    tp = np.diag(cm)
    p = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)     # column = predicted
    r = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)     # row = gold
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    mp, mr = float(p.mean()), float(r.mean())
    return {"n": int(cm.sum()), "macro_precision": mp, "macro_recall": mr, "macro_f1": float(f.mean()),
            "harmonic_mean_macro_p_r": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
            "per_class": [{"precision": float(a), "recall": float(b), "f1": float(c), "support": int(s)}
                          for a, b, c, s in zip(p, r, f, cm.sum(axis=1))]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrices", required=True)
    ap.add_argument("--reported-four", type=float, default=None, help="originally reported four-class F1 (0.773)")
    ap.add_argument("--reported-binary", type=float, default=None, help="originally reported binary F1 (0.878)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.matrices, encoding="utf-8") as f:
        mats = json.load(f)
    rows = []
    for key, name, rep in (("four_class", "Four-class", args.reported_four), ("binary", "Binary", args.reported_binary)):
        if key in mats:
            m = metrics_from_confusion(mats[key])
            rows.append((name, m, rep))
    md = ["**Table R2 — Recomputation of the originally reported values**", "",
          "| Setting | Predictions | Macro precision | Macro recall | Macro-F1 (mean of per-class F1) | "
          "Harmonic mean of macro P and R | Originally reported |", "|---|---|---|---|---|---|---|"]
    for name, m, rep in rows:
        md.append(f"| {name} | {m['n']} | {m['macro_precision']:.3f} | {m['macro_recall']:.3f} | {m['macro_f1']:.3f} | "
                  f"{m['harmonic_mean_macro_p_r']:.3f} | {'' if rep is None else f'{rep:.3f}'} |")
    text = "\n".join(md) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out + ".md", "w", encoding="utf-8") as f:
            f.write(text)
        with open(args.out + ".json", "w", encoding="utf-8") as f:
            json.dump({n: m for n, m, _ in rows}, f, indent=2)


if __name__ == "__main__":
    main()
