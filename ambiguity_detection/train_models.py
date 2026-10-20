
# python train_models.py --model siamese --task binary --seed 1 \
#     --encoder bert-base-uncased --data-dir data --out-dir results

from __future__ import annotations

import argparse
import os
import platform
import random
import time

import numpy as np
import pandas as pd

from cn_common import (TASKS, compute_metrics, data_summary, load_splits, macro_f1, set_seed, task_labels,
                       write_json, write_predictions)


# =========================================================================== classical baselines
def run_majority(splits, task, seed):
    ytr = task_labels(splits["train"]["level"], task)
    vals, counts = np.unique(ytr, return_counts=True)
    maj = int(vals[np.argmax(counts)])
    preds = {k: np.full(len(splits[k]), maj) for k in ("dev", "test")}
    return preds, {"majority_class": maj}


def run_tfidf_lr(splits, task, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1, sublinear_tf=True)
    Xtr = vec.fit_transform(splits["train"]["initial_request"])
    Xdev, Xte = vec.transform(splits["dev"]["initial_request"]), vec.transform(splits["test"]["initial_request"])
    ytr, ydev = task_labels(splits["train"]["level"], task), task_labels(splits["dev"]["level"], task)
    best = None
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed).fit(Xtr, ytr)
        score = macro_f1(ydev, clf.predict(Xdev), task)
        if best is None or score > best[0]:
            best = (score, C, clf)
    _, C, clf = best
    return {"dev": clf.predict(Xdev), "test": clf.predict(Xte)}, {"selected_C": C, "dev_macro_f1": best[0]}


# =========================================================================== neural models
def _torch_setup(args):
    import torch
    torch.set_num_threads(args.threads or os.cpu_count() or 1)
    dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    return torch, dev


def _class_weights(torch, y, n_classes, device):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = len(y) / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float, device=device)


def run_siamese(splits, task, seed, args):
    torch, device = _torch_setup(args)
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    labels = TASKS[task]["labels"]
    to_idx = {l: i for i, l in enumerate(labels)}
    texts = {k: splits[k]["initial_request"].tolist() for k in splits}
    y = {k: np.array([to_idx[v] for v in task_labels(splits[k]["level"], task)]) for k in splits}

    tok = AutoTokenizer.from_pretrained(args.encoder)

    class Encoder(nn.Module):
        """Transformer + mean pooling + shared MLP (ReLU) projection, L2-normalised (Fig. 1)."""

        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(args.encoder)
            h = self.backbone.config.hidden_size
            self.proj = nn.Sequential(nn.Linear(h, args.proj_hidden), nn.ReLU(), nn.Linear(args.proj_hidden, args.proj_dim))

        def forward(self, batch_texts):
            enc = tok(batch_texts, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
            out = self.backbone(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            return F.normalize(self.proj(pooled), dim=-1)

    enc = Encoder().to(device)

    # ---- Phase 1: triplet loss, Eq. (1): max(||A-P||^2 - ||A-N||^2 + alpha, 0)
    # Anchor class is drawn uniformly over classes, so minority classes get as many triplets as majority ones.
    by_class = {c: np.where(y["train"] == c)[0].tolist() for c in range(len(labels))}
    usable = [c for c, ix in by_class.items() if len(ix) >= 2]
    rng = random.Random(seed)
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr)
    t0 = time.time()
    triplet_log = []
    for ep in range(args.triplet_epochs):
        enc.train()
        losses = []
        for _ in range(max(1, args.triplets_per_epoch // args.batch_size)):
            A, P, N = [], [], []
            for _ in range(args.batch_size):
                c = rng.choice(usable)
                a, p = rng.sample(by_class[c], 2)
                n = rng.choice(by_class[rng.choice([k for k in by_class if k != c and by_class[k]])])
                A.append(texts["train"][a]); P.append(texts["train"][p]); N.append(texts["train"][n])
            ea, ep_, en = enc(A), enc(P), enc(N)
            loss = F.relu((ea - ep_).pow(2).sum(-1) - (ea - en).pow(2).sum(-1) + args.margin).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        triplet_log.append(float(np.mean(losses)))
        print(f"  [siamese seed={seed}] triplet epoch {ep + 1}/{args.triplet_epochs} loss={triplet_log[-1]:.4f}", flush=True)

    # ---- Phase 2: frozen encoder -> embeddings -> MLP head (class-weighted CE), select epoch on dev macro-F1
    enc.eval()

    @torch.no_grad()
    def embed(ts):
        return torch.cat([enc(ts[i:i + 64]) for i in range(0, len(ts), 64)])

    E = {k: embed(texts[k]) for k in texts}
    head = nn.Sequential(nn.Linear(args.proj_dim, args.head_hidden), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(args.head_hidden, len(labels))).to(device)
    hopt = torch.optim.AdamW(head.parameters(), lr=args.head_lr)
    ce = nn.CrossEntropyLoss(weight=_class_weights(torch, y["train"], len(labels), device))
    ytr_t = torch.tensor(y["train"], device=device)
    g = torch.Generator().manual_seed(seed)
    best = (-1.0, None, -1)
    for ep in range(args.head_epochs):
        head.train()
        perm = torch.randperm(len(ytr_t), generator=g)
        for i in range(0, len(perm), args.batch_size):
            ix = perm[i:i + args.batch_size]
            loss = ce(head(E["train"][ix]), ytr_t[ix])
            hopt.zero_grad(); loss.backward(); hopt.step()
        head.eval()
        with torch.no_grad():
            pdev = head(E["dev"]).argmax(-1).cpu().numpy()
        s = macro_f1(np.array(labels)[y["dev"]], np.array(labels)[pdev], task)
        if s > best[0]:
            best = (s, {k: v.detach().clone() for k, v in head.state_dict().items()}, ep + 1)
    head.load_state_dict(best[1]); head.eval()
    with torch.no_grad():
        preds = {k: np.array(labels)[head(E[k]).argmax(-1).cpu().numpy()] for k in ("dev", "test")}
    info = {"triplet_loss_per_epoch": triplet_log, "selected_head_epoch": best[2], "dev_macro_f1": best[0],
            "train_seconds": round(time.time() - t0, 1),
            "n_parameters": int(sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in head.parameters()))}
    return preds, info


def run_finetune(splits, task, seed, args):
    torch, device = _torch_setup(args)
    import torch.nn as nn
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    labels = TASKS[task]["labels"]
    to_idx = {l: i for i, l in enumerate(labels)}
    texts = {k: splits[k]["initial_request"].tolist() for k in splits}
    y = {k: np.array([to_idx[v] for v in task_labels(splits[k]["level"], task)]) for k in splits}

    tok = AutoTokenizer.from_pretrained(args.encoder)
    model = AutoModelForSequenceClassification.from_pretrained(args.encoder, num_labels=len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(weight=_class_weights(torch, y["train"], len(labels), device))
    g = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def predict(ts):
        model.eval()
        out = []
        for i in range(0, len(ts), 64):
            e = tok(ts[i:i + 64], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
            out.append(model(**e).logits.argmax(-1).cpu().numpy())
        return np.concatenate(out)

    t0 = time.time()
    best = (-1.0, None, -1)
    for ep in range(args.ft_epochs):
        model.train()
        perm = torch.randperm(len(y["train"]), generator=g).tolist()
        for i in range(0, len(perm), args.batch_size):
            ix = perm[i:i + args.batch_size]
            e = tok([texts["train"][j] for j in ix], padding=True, truncation=True, max_length=args.max_len,
                    return_tensors="pt").to(device)
            loss = ce(model(**e).logits, torch.tensor(y["train"][ix], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
        s = macro_f1(np.array(labels)[y["dev"]], np.array(labels)[predict(texts["dev"])], task)
        print(f"  [finetune seed={seed}] epoch {ep + 1}/{args.ft_epochs} dev macro-F1={s:.4f}", flush=True)
        if s > best[0]:
            best = (s, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    preds = {k: np.array(labels)[predict(texts[k])] for k in ("dev", "test")}
    info = {"selected_epoch": best[2], "dev_macro_f1": best[0], "train_seconds": round(time.time() - t0, 1),
            "n_parameters": int(sum(p.numel() for p in model.parameters()))}
    return preds, info


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["siamese", "finetune", "tfidf_lr", "majority"])
    ap.add_argument("--name", default=None, help="Name of the run folder (default: model name, e.g. siamese_bert)")
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--encoder", default="bert-base-uncased",
                    help="Hugging Face id OR local folder (e.g. models/bert-base-uncased)")
    ap.add_argument("--max-len", type=int, default=32, help="ClariQ requests are <= 16 words")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    # siamese
    ap.add_argument("--margin", type=float, default=0.5, help="alpha in Eq. (1)")
    ap.add_argument("--triplet-epochs", type=int, default=8)
    ap.add_argument("--triplets-per-epoch", type=int, default=256)
    ap.add_argument("--proj-hidden", type=int, default=256)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--head-epochs", type=int, default=100)
    # finetune
    ap.add_argument("--ft-epochs", type=int, default=8)
    # hardware
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    splits = load_splits(args.data_dir)
    write_json(data_summary(splits), os.path.join(args.out_dir, "data_summary.json"))

    name = args.name or args.model
    run_dir = os.path.join(args.out_dir, args.task, name, f"seed{args.seed}")
    print(f"== {name} | task={args.task} | seed={args.seed} | train/dev/test topics = "
          f"{len(splits['train'])}/{len(splits['dev'])}/{len(splits['test'])}", flush=True)

    t0 = time.time()
    if args.model == "majority":
        preds, info = run_majority(splits, args.task, args.seed)
    elif args.model == "tfidf_lr":
        preds, info = run_tfidf_lr(splits, args.task, args.seed)
    elif args.model == "siamese":
        preds, info = run_siamese(splits, args.task, args.seed, args)
    else:
        preds, info = run_finetune(splits, args.task, args.seed, args)

    metrics = {}
    for split in ("dev", "test"):
        yt = task_labels(splits[split]["level"], args.task)
        write_predictions(splits[split], yt, preds[split], args.task, os.path.join(run_dir, f"predictions_{split}.csv"))
        metrics[split] = compute_metrics(yt, preds[split], args.task)

    # Official ClariQ run file (topic_id <space> level), evaluable with ClariQ's src/clariq_eval_tool.py
    if args.task == "four":
        with open(os.path.join(run_dir, "clariq_run_test.txt"), "w") as f:
            for tid, p in zip(splits["test"]["topic_id"], preds["test"]):
                f.write(f"{tid} {int(p)}\n")

    import sklearn
    env = {"python": platform.python_version(), "platform": platform.platform(), "sklearn": sklearn.__version__}
    try:
        import torch, transformers
        env.update(torch=torch.__version__, transformers=transformers.__version__,
                   device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    except ImportError:
        pass
    write_json({"args": vars(args), "info": info, "environment": env, "wall_seconds": round(time.time() - t0, 1),
                "metrics": metrics}, os.path.join(run_dir, "run_info.json"))
    t = metrics["test"]
    print(f"   TEST macro-F1={t['macro_f1']:.4f} weighted-F1={t['weighted_f1']:.4f} acc={t['accuracy']:.4f} "
          f"({time.time() - t0:.0f}s) -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
