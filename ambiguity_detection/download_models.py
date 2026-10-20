
#Download bert-base-uncased and roberta-base once into ./models so every

import argparse
import os
import shutil
import ssl
import sys
import urllib.request

MODELS = {
    "bert-base-uncased": ["config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json", "model.safetensors"],
    "roberta-base": ["config.json", "vocab.json", "merges.txt", "tokenizer.json", "model.safetensors"],
}


def fetch(url, dst):
    tmp = dst + ".part"
    ctx = None
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
                sys.stdout.flush()
    if total:
        print()
    shutil.move(tmp, dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", action="store_true", help="download from https://hf-mirror.com")
    a = ap.parse_args()
    base = "https://hf-mirror.com" if a.mirror else os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    here = os.path.dirname(os.path.abspath(__file__))
    for repo, files in MODELS.items():
        dst_dir = os.path.join(here, "models", repo)
        os.makedirs(dst_dir, exist_ok=True)
        print(f"{repo} -> {dst_dir}")
        for fn in files:
            dst = os.path.join(dst_dir, fn)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                print(f"  {fn}: already there, skipped")
                continue
            url = f"{base}/{repo}/resolve/main/{fn}"
            print(f"  {fn}  <-  {url}")
            fetch(url, dst)
    print("Done. run_all.py will use these local copies automatically.")
