"""
Pulls Bach/Beethoven/Chopin/Mozart out of the full Kaggle midi-classic-music
dump and drops them into data/raw/<composer>/ with clean filenames.

Run this after downloading the dataset with:
    kaggle datasets download -d blanderbuss/midi-classic-music -p data_raw --unzip

Usage:
    python scripts/filter_dataset.py
"""

import csv
import shutil
import zipfile
from pathlib import Path

RAW_DIR = Path("data_raw/midiclassics")
OUT_DIR = Path("data/raw")

# folder names in the dataset don't always match how we want to label things
COMPOSERS = {
    "Bach": "bach",
    "Beethoven": "beethoven",
    "Chopin": "chopin",
    "Mozart": "mozart",
}


def unzip_nested(composer_dir):
    # a handful of Beethoven pieces are shipped as zip files instead of
    # raw .mid, so unzip them in place before we go looking for midi files
    for zip_path in composer_dir.rglob("*.zip"):
        extract_to = zip_path.with_suffix("")
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(extract_to)
        except zipfile.BadZipFile:
            print(f"  skipping bad zip: {zip_path.name}")


def clean_name(path, used_names):
    name = path.stem.strip().replace(" ", "_")
    # strip out characters that are annoying on windows paths
    name = "".join(c for c in name if c.isalnum() or c in "_-")
    if not name:
        name = "untitled"
    candidate = f"{name}.mid"
    i = 2
    while candidate in used_names:
        candidate = f"{name}_{i}.mid"
        i += 1
    used_names.add(candidate)
    return candidate


def main():
    if not RAW_DIR.exists():
        raise SystemExit(f"can't find {RAW_DIR}, did you download+unzip the kaggle dataset first?")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for folder_name, label in COMPOSERS.items():
        src = RAW_DIR / folder_name
        if not src.exists():
            print(f"WARNING: no folder for {folder_name}, skipping")
            continue

        unzip_nested(src)

        dest = OUT_DIR / label
        dest.mkdir(parents=True, exist_ok=True)

        used_names = set()
        count = 0
        for midi_path in sorted(src.rglob("*.mid")):
            out_name = clean_name(midi_path, used_names)
            shutil.copy2(midi_path, dest / out_name)
            manifest_rows.append([label, out_name, str(midi_path.relative_to(RAW_DIR))])
            count += 1

        print(f"{label}: copied {count} files")

    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["composer", "filename", "source_path"])
        writer.writerows(manifest_rows)

    print(f"\nwrote manifest with {len(manifest_rows)} total files to {manifest_path}")


if __name__ == "__main__":
    main()
