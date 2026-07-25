# Data Preprocessing Notes

Quick writeup of how `data/raw/` got built, mostly so I remember what I did and so the report section isn't written from memory later.

## Source

[MIDI Classic Music](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music) on Kaggle, uploaded by user blanderbuss. It's a scrape of classical MIDI files organized into one folder per composer, 136 composers total, about 145 MB unzipped.

We only need 4 of those composers per the project instructions, so most of the dataset gets ignored.

## What the filter script does

`scripts/filter_dataset.py` does the actual work. Steps:

1. Look at the `Bach`, `Beethoven`, `Chopin`, `Mozart` folders inside the downloaded dataset.
2. A few Beethoven pieces are shipped as `.zip` files instead of raw `.mid` (looks like multi-movement sonatas got zipped up together). The script unzips those in place first so they don't get skipped.
3. Walk each composer folder recursively and grab every `.mid` file. Some folders have nested subfolders (like Bach's 400 chorales), so this has to recurse, not just look one level deep.
4. Copy each file into `data/raw/<composer>/` using a lowercase label (`bach`, `beethoven`, `chopin`, `mozart`) and a cleaned up filename, since the original filenames have spaces, apostrophes, and other characters that are annoying to deal with later.
5. If two files would end up with the same cleaned name, the script appends a number so nothing gets overwritten.
6. Write everything to `data/raw/manifest.csv` with columns `composer, filename, source_path`, so we can always trace a file back to where it came from in the original dump.

## Things worth noting

- There's one non-MIDI file sitting in the Bach folder (a plain text `readme` inside the 400 Chorales subfolder). The script only grabs `*.mid` files so it gets skipped automatically, no manual cleanup needed.
- The classes are not balanced. Bach has over 1000 files, Chopin only has 136. This is going to matter for training, probably need class weighting or augmentation so the models don't just learn to guess "Bach" all the time.
- File counts after filtering:

| Composer | Files | Size |
|---|---|---|
| Bach | 1024 | ~13.2 MB |
| Beethoven | 220 | ~14.0 MB |
| Chopin | 136 | ~2.8 MB |
| Mozart | 257 | ~10.6 MB |

## Next steps

- Parse each file into note/pitch/duration/velocity sequences for the LSTM track.
- Convert each file into a piano roll matrix for the CNN track.
- Split into train/val/test, same split used by both tracks so results are comparable.
- Figure out augmentation to help with the class imbalance (probably pitch shifting and/or time stretching).
