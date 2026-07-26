# ComposerPrediction

AAI-511 final project, Team 5. We're building two deep learning models that guess which composer wrote a piece of classical music: Bach, Beethoven, Chopin, or Mozart. One model is an LSTM that reads note sequences. The other is a CNN that reads piano roll images. At the end we compare how well each one does.

## Team

- Christina Sadiq
- Jackson Kenyon
- Dylan Scott-Dawkins

### Who built what

| Area | Owner |
|---|---|
| Dataset filtering (`scripts/filter_dataset.py`), `docs/data_preprocessing.md` | Christina Sadiq |
| MIDI parsing to note sequences (`02_parse_midi_lstm.ipynb`) | Christina Sadiq |
| Piano roll conversion (`03_piano_roll_cnn.ipynb`) | Christina Sadiq |
| Train/val/test split and augmentation (`04_split_and_augment.ipynb`) | Christina Sadiq |
| Repo scaffold | Dylan Scott-Dawkins |
| LSTM model (`05_lstm_model.ipynb`) | Dylan Scott-Dawkins |
| CNN model (`06_cnn_model.ipynb`) | Dylan Scott-Dawkins |
| Model comparison (`07_compare.ipynb`), `docs/interface-contract.md` | Dylan Scott-Dawkins |

Everything in notebooks 05–07 consumes Christina's preprocessing output as-is — the `.npz` sequences, the `.npy` piano rolls, and the split CSVs from notebook 04 are used unmodified, including her decisions on the 70/15/15 stratified split, pitch-shift augmentation on the training set only, and keeping the real composer distribution in val and test.

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
│   ├── processed/
│   │   ├── lstm/             # per-piece .npz: pitch, duration, velocity, offset
│   │   └── cnn/              # per-piece .npy: 88 x time piano roll
│   └── splits/               # train.csv (augmented), train_original.csv, val.csv, test.csv
├── notebooks/
│   ├── 02_parse_midi_lstm.ipynb    # MIDI -> note sequences
│   ├── 03_piano_roll_cnn.ipynb     # MIDI -> piano rolls
│   ├── 04_split_and_augment.ipynb  # stratified split + pitch-shift augmentation
│   ├── 05_lstm_model.ipynb         # LSTM: train + evaluate
│   ├── 06_cnn_model.ipynb          # CNN: train + evaluate
│   └── 07_compare.ipynb            # side-by-side comparison
├── models/                   # trained .keras weights (gitignored, regenerate with 05/06)
├── results/                  # metrics JSON, comparison.csv, figures
├── scripts/
│   └── filter_dataset.py     # pulls our 4 composers out of the full dataset
├── docs/
│   ├── data_preprocessing.md # notes on how the dataset was built
│   └── interface-contract.md # shapes and files each model step reads/writes
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
- [x] MIDI files parsed into note sequences for the LSTM
- [x] MIDI files converted to piano roll images for the CNN
- [x] Train/val/test split
- [ ] LSTM model — written, tuning in progress
- [ ] CNN model — written, tuning in progress
- [ ] Model comparison
- [ ] Final report and notebook
