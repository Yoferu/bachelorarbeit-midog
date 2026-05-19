import argparse
import json
from pathlib import Path

import pandas as pd


def parse_slide_id(filename: str) -> int:
    """
    Converts '001.tiff' or '1.tiff' to 1.
    """
    return int(Path(str(filename)).stem)


def midpoint_from_bbox(bbox, bbox_format: str):
    """
    The MIDOG_2025_Guide notebook uses:
        x = int((bbox[0] + bbox[2]) / 2)
        y = int((bbox[1] + bbox[3]) / 2)

    Therefore default is 'xyxy':
        bbox = [xmin, ymin, xmax, ymax]

    If your local JSON uses standard COCO bbox format:
        bbox = [xmin, ymin, width, height]
    then call this script with --bbox-format xywh.
    """
    if bbox_format == "xyxy":
        x = int((bbox[0] + bbox[2]) / 2)
        y = int((bbox[1] + bbox[3]) / 2)
    elif bbox_format == "xywh":
        x = int(bbox[0] + bbox[2] / 2)
        y = int(bbox[1] + bbox[3] / 2)
    else:
        raise ValueError(f"Unsupported bbox format: {bbox_format}")

    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--midog-json",
        required=True,
        help="Path to MIDOGpp.json"
    )
    parser.add_argument(
        "--xvalidation",
        required=True,
        help="Path to datasets_xvalidation.csv from MIDOGpp repo"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path for MIDOG_2025_Guide/evaluate.py"
    )
    parser.add_argument(
        "--bbox-format",
        choices=["xyxy", "xywh"],
        default="xyxy",
        help="Use xyxy to match MIDOG_2025_Guide notebook logic; use xywh only if your JSON bbox is standard COCO."
    )
    args = parser.parse_args()

    midog_json = Path(args.midog_json)
    xvalidation_csv = Path(args.xvalidation)
    out_csv = Path(args.out)

    with midog_json.open("r", encoding="utf-8") as f:
        database = json.load(f)

    # Read MIDOG++ official train/test split.
    split_df = pd.read_csv(xvalidation_csv, sep=";")
    split_df["Slide"] = split_df["Slide"].astype(int)

    split_map = dict(zip(split_df["Slide"], split_df["Dataset"]))
    tumor_map = dict(zip(split_df["Slide"], split_df["Tumor"]))

    # Image table.
    image_df = pd.DataFrame.from_dict(database["images"])
    image_df = image_df.rename(
        columns={
            "id": "image_id",
            "file_name": "filename",
            "tumor_type": "tumortype",
        }
    )

    image_df["filename"] = image_df["filename"].apply(lambda p: Path(str(p)).name)
    image_df["slide"] = image_df["filename"].apply(parse_slide_id)

    image_df["split"] = image_df["slide"].map(split_map)
    image_df["tumortype_from_split"] = image_df["slide"].map(tumor_map)

    if "tumortype" not in image_df.columns:
        image_df["tumortype"] = image_df["tumortype_from_split"]
    else:
        image_df["tumortype"] = image_df["tumortype"].fillna(image_df["tumortype_from_split"])

    # Annotation table.
    annotations_df = pd.DataFrame.from_dict(database["annotations"])

    xs = []
    ys = []
    for bbox in annotations_df["bbox"]:
        x, y = midpoint_from_bbox(bbox, args.bbox_format)
        xs.append(x)
        ys.append(y)

    annotations_df["x"] = xs
    annotations_df["y"] = ys

    annotations_df = annotations_df.rename(columns={"category_id": "label"})

    # Keep only the columns needed for merge.
    annotations_df = annotations_df[["image_id", "x", "y", "label"]]

    # Merge annotations with image metadata and MIDOG++ split.
    comb_df = annotations_df.merge(
        image_df[["image_id", "filename", "slide", "split", "tumortype"]],
        how="left",
        on="image_id",
    )

    missing_split = comb_df["split"].isna().sum()
    if missing_split:
        missing_slides = sorted(comb_df.loc[comb_df["split"].isna(), "slide"].dropna().unique().tolist())
        print(f"Warning: {missing_split} annotations have no split mapping.")
        print(f"First missing slide IDs: {missing_slides[:20]}")
        comb_df = comb_df.dropna(subset=["split"])

    comb_df = comb_df[["x", "y", "label", "filename", "slide", "split", "tumortype"]]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    comb_df.to_csv(out_csv, index=False)

    print(f"Wrote: {out_csv}")
    print()
    print("Rows:", len(comb_df))
    print("Images:", comb_df["filename"].nunique())
    print()
    print("Images per split:")
    print(comb_df.groupby("split")["filename"].nunique())
    print()
    print("Annotations per split/label:")
    print(comb_df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()
