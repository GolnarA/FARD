
# Download the three official ClariQ files used for the clarification-need task.
# python download_data.py --out data



import argparse
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/aliannejadi/ClariQ/master/data/"
FILES = ["train.tsv", "dev.tsv", "test_with_labels.tsv"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in FILES:
        dst = os.path.join(a.out, f)
        print(f"downloading {BASE + f} -> {dst}")
        urllib.request.urlretrieve(BASE + f, dst)
        print(f"  ok ({os.path.getsize(dst) / 1e6:.1f} MB)")
    print("Done.")
