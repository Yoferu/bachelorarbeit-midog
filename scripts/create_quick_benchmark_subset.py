import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_STRATIFY_COLUMNS = ("tumortype", "scanner", "Scanner", "source", "dataset")


def _annotation_count_bins(counts: pd.Series) -> pd.Series:
    labels = pd.Series("few_mitoses", index=counts.index, dtype="object")
    labels[counts == 0] = "no_mitoses"

    positive = counts[counts > 0]
    if positive.nunique() > 1:
        median_positive = positive.median()
        labels[counts > median_positive] = "many_mitoses"

    return labels


def _proportional_sample(groups: pd.core.groupby.DataFrameGroupBy, n: int, seed: int) -> pd.Index:
    shuffled_index = groups.obj.sample(frac=1.0, random_state=seed).index
    group_sizes = groups.size()
    total = int(group_sizes.sum())

    quotas = (group_sizes / total * n).apply(int)
    quotas[group_sizes > 0] = quotas[group_sizes > 0].clip(lower=1)

    while int(quotas.sum()) > n:
        reducible = quotas[quotas > 1].index
        if len(reducible) == 0:
            break
        largest = (quotas.loc[reducible] - (group_sizes.loc[reducible] / total * n)).idxmax()
        quotas.loc[largest] -= 1

    while int(quotas.sum()) < n:
        remaining = group_sizes - quotas
        remaining = remaining[remaining > 0]
        if remaining.empty:
            break
        quotas.loc[remaining.idxmax()] += 1

    selected = []
    for key, group in groups:
        quota = min(int(quotas.loc[key]), len(group))
        if quota <= 0:
            continue
        selected.extend(group.sample(n=quota, random_state=seed).index.tolist())

    if len(selected) < n:
        missing = n - len(selected)
        leftovers = groups.obj.drop(index=selected)
        if not leftovers.empty:
            selected.extend(leftovers.sample(n=min(missing, len(leftovers)), random_state=seed).index.tolist())

    selected_set = set(selected[:n])
    return pd.Index([idx for idx in shuffled_index if idx in selected_set])


def create_subset(
    input_csv: Path,
    output_csv: Path,
    n: int,
    seed: int,
    split: str,
    stratify_auto: bool,
    metadata_json: Path | None,
) -> None:
    df = pd.read_csv(input_csv)
    if "split" not in df.columns:
        raise ValueError("Input CSV must contain a 'split' column.")
    if "filename" not in df.columns:
        raise ValueError("Input CSV must contain a 'filename' column.")

    split_df = df.query("split == @split").copy()
    if split_df.empty:
        raise ValueError(f"No rows found for split={split!r}.")

    image_rows = split_df.drop_duplicates("filename").set_index("filename", drop=False)

    if "label" in split_df.columns:
        mitosis_counts = split_df.query("label == 1").groupby("filename").size()
    else:
        mitosis_counts = pd.Series(dtype="int64")
    image_rows["mitosis_count"] = image_rows.index.map(mitosis_counts).fillna(0).astype(int)
    image_rows["mitosis_bin"] = _annotation_count_bins(image_rows["mitosis_count"])

    n = min(n, len(image_rows))
    rng_seed = seed

    if stratify_auto:
        stratify_cols = [col for col in DEFAULT_STRATIFY_COLUMNS if col in image_rows.columns]
        stratify_cols.append("mitosis_bin")
        sampled_index = _proportional_sample(image_rows.groupby(stratify_cols, dropna=False), n, rng_seed)
    else:
        sampled_index = image_rows.sample(n=n, random_state=rng_seed).index

    selected_filenames = image_rows.loc[sampled_index, "filename"].tolist()
    selected = split_df[split_df["filename"].isin(selected_filenames)].copy()

    order = {filename: idx for idx, filename in enumerate(selected_filenames)}
    selected["_quick_order"] = selected["filename"].map(order)
    selected = selected.sort_values(["_quick_order"]).drop(columns=["_quick_order"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False)

    metadata = {
        "description": "Fixed quick benchmark subset generated from the official test split only.",
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "split": split,
        "seed": seed,
        "requested_images": n,
        "selected_images": len(selected_filenames),
        "selected_rows": len(selected),
        "stratify_auto": stratify_auto,
        "columns_preserved": list(df.columns),
        "selected_filenames": selected_filenames,
        "mitosis_count_summary": image_rows.loc[selected_filenames, "mitosis_count"].describe().to_dict(),
    }

    metadata_path = metadata_json or output_csv.with_suffix(".meta.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote quick benchmark CSV: {output_csv}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Selected images: {len(selected_filenames)}")
    print(f"Selected rows: {len(selected)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed quick benchmark subset from an official split CSV.")
    parser.add_argument("--input_csv", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--n", type=int, default=32, help="Number of images to select.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--stratify_auto", action="store_true")
    parser.add_argument("--metadata_json", type=Path, default=None)
    args = parser.parse_args()

    create_subset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        n=args.n,
        seed=args.seed,
        split=args.split,
        stratify_auto=args.stratify_auto,
        metadata_json=args.metadata_json,
    )


if __name__ == "__main__":
    main()
