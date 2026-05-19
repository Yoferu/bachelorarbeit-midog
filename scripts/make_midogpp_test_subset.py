from pathlib import Path
import argparse
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-images", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--out-images", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    raw_images = Path(args.raw_images)
    out_images = Path(args.out_images)
    out_images.mkdir(parents=True, exist_ok=True)

    with open(args.coco_json, "r") as f:
        coco = json.load(f)

    images = sorted(coco["images"], key=lambda x: x.get("file_name", ""))

    selected = []
    for img in images:
        file_name = img["file_name"]
        src = raw_images / file_name
        if src.exists():
            selected.append(img)
        if len(selected) >= args.limit:
            break

    if not selected:
        raise RuntimeError("No selected images found. Check file_name values and image directory.")

    selected_ids = {img["id"] for img in selected}

    subset = {
        "images": selected,
        "annotations": [
            ann for ann in coco["annotations"]
            if ann["image_id"] in selected_ids
        ],
        "categories": coco.get("categories", []),
    }

    for img in selected:
        src = raw_images / img["file_name"]
        dst = out_images / img["file_name"]

        if dst.exists():
            continue

        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)

    with open(args.out_json, "w") as f:
        json.dump(subset, f, indent=2)

    print(f"Selected {len(selected)} images")
    print(f"Selected {len(subset['annotations'])} annotations")
    print(f"Images written to: {out_images}")
    print(f"Annotation subset written to: {args.out_json}")


if __name__ == "__main__":
    main()
