# src/filter_subreddits.py
import json
from pathlib import Path
from typing import Iterable, Dict, Any, List

import zstandard as zstd
import pandas as pd

RAW_DIR   = Path("data/raw/reddit/subreddits24")
OUT_DIR   = Path("data/processed/submissions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# choose which inputs to process now (add WSB later, it’s big)
TARGETS = [
    "finance_submissions.zst",
    "stocks_submissions.zst",
    "investing_submissions.zst",
    "StockMarket_submissions.zst",
     "wallstreetbets_submissions.zst",  # add later if you want
]

KEEP = ("id","subreddit","created_utc","title","selftext","score","num_comments","author","url","permalink")
BATCH_SIZE = 50_000
MAX_ROWS = None   # set to an int (e.g. 200_000) to sample while testing

def stream_jsonl_from_zst(zst_path: Path) -> Iterable[Dict[str, Any]]:
    with zst_path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while True:
                chunk = reader.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                buf += chunk
                lines = buf.split(b"\n")
                buf = lines.pop()
                for line in lines:
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue

def project_row(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {k: obj.get(k) for k in KEEP}

def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce").astype("Int64")
    for col in ["id","subreddit","title","selftext","author","url","permalink"]:
        if col in df:
            df[col] = df[col].astype("string")
    for col in ["score","num_comments"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def convert_one(zst_path: Path):
    print(f"\n>> Reading: {zst_path.resolve()}")
    assert zst_path.exists(), f"Input not found: {zst_path}"
    part = 0
    total = 0
    batch: List[Dict[str, Any]] = []

    for i, obj in enumerate(stream_jsonl_from_zst(zst_path), start=1):
        if i % 10_000 == 0:
            print(f"[{zst_path.name}] ... parsed {i:,} lines")
        batch.append(project_row(obj))

        if len(batch) >= BATCH_SIZE:
            part += 1
            out_path = OUT_DIR / f"{zst_path.stem}_part{part:03d}.parquet"
            df = normalize_dtypes(pd.DataFrame(batch))
            df.to_parquet(out_path, index=False)
            total += len(batch)
            batch.clear()

        if MAX_ROWS and i >= MAX_ROWS:
            break

    if batch:
        part += 1
        out_path = OUT_DIR / f"{zst_path.stem}_part{part:03d}.parquet"
        df = normalize_dtypes(pd.DataFrame(batch))
        df.to_parquet(out_path, index=False)
        total += len(batch)

    print(f"[{zst_path.name}] Done. Wrote {part} parquet part(s), rows ≈ {total:,}")

def main():
    print("Planned inputs:")
    inputs = [RAW_DIR / f for f in TARGETS]
    for p in inputs:
        print(" -", p)
    for p in inputs:
        convert_one(p)

if __name__ == "__main__":
    main()