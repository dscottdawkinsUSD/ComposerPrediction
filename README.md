# ComposerPrediction

AAI-511 final project, Team 5. We're building two deep learning models that guess which composer wrote a piece of classical music: Bach, Beethoven, Chopin, or Mozart. One model is an LSTM that reads note sequences. The other is a CNN that reads piano roll images. At the end we compare how well each one does.

## Team

- Christina Sadiq
- Jackson Kenyon
- Dylan Scott-Dawkins

## Dataset

We're using the [MIDI Classic Music dataset](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music) from Kaggle. It has MIDI files from over 100 composers, but we only need four: Bach, Beethoven, Chopin, Mozart.

The raw download is way bigger than we need (136 composer folders, 4700+ files), so `scripts/filter_dataset.py` pulls out just our four composers and copies them into `data/raw/<composer>/` with clean filenames. It also unzips the handful of Beethoven pieces that came as zip files instead of plain .mid files, and writes a manifest so we know where every file came from.

Current counts after filtering:

| Composer | Files |
|---|---|
| Bach | 1024 |
| Beethoven | 220 |
| Chopin | 136 |
| Mozart | 257 |

Total: 1637 MIDI files, about 41 MB.

## Repo structure

```
ComposerPrediction/
├── data/
│   └── raw/
│       ├── bach/
│       ├── beethoven/
│       ├── chopin/
│       ├── mozart/
│       └── manifest.csv      # composer, filename, original path in the kaggle dump
├── scripts/
│   └── filter_dataset.py     # pulls our 4 composers out of the full dataset
├── docs/
│   └── data_preprocessing.md # notes on how the dataset was built
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo.
2. Make a virtual environment and install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Get a Kaggle API key (`kaggle.json`) and drop it in `~/.kaggle/`. Instructions are in the [Kaggle API docs](https://www.kaggle.com/docs/api).
4. Download the raw dataset:
   ```
   kaggle datasets download -d blanderbuss/midi-classic-music -p data_raw --unzip
   ```
5. Run the filter script from the repo root:
   ```
   python scripts/filter_dataset.py
   ```
   This builds `data/raw/` from `data_raw/`. `data_raw/` itself is gitignored since it's 145 MB and we don't need most of it.

## Status

- [x] Dataset downloaded and filtered to our 4 composers
- [ ] MIDI files parsed into note sequences for the LSTM
- [ ] MIDI files converted to piano roll images for the CNN
- [ ] Train/val/test split
- [ ] LSTM model
- [ ] CNN model
- [ ] Model comparison
- [ ] Final report and notebook
