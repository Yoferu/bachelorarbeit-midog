import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []

    for path in sorted(Path(args.results_dir).glob("*_metrics.json")):
        with path.open("r") as f:
            data = json.load(f)

        agg = data.get("aggregates", {})
        rows.append({
            "run": path.stem.replace("_metrics", ""),
            "precision": agg.get("precision"),
            "recall": agg.get("recall"),
            "f1_score": agg.get("f1_score"),
            "AP": agg.get("AP"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
