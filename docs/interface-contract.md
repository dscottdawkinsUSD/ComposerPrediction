# Interface Contract — LSTM model (notebook 05)

Inputs already exist and are unchanged. This doc defines only what the LSTM model step reads and writes.

The input side — the `.npz` sequences, the `.npy` piano rolls, the manifests and the split CSVs — is Christina Sadiq's preprocessing pipeline (notebooks 02–04, commit `60c0866`), transcribed here as-is. Notebook 05 reads it and does not modify it.

The CNN track and the model comparison are not covered here yet; those interfaces will be added when those notebooks land.

## Interface: LSTM sample (read)
Source: `data/processed/lstm/<composer>/<stem>.npz`
Direction: input

| Field | Type | Required | Description | Example |
|---|---|---|---|---|
| pitch | int array (N,) | yes | MIDI pitch per note event | [60, 62, 64] |
| duration | float array (N,) | yes | Note length in seconds | [0.5, 0.25] |
| velocity | int array (N,) | yes | MIDI velocity 0–127 | [80, 64] |
| offset | float array (N,) | yes | Absolute start time in seconds | [0.0, 0.5] |

Model tensors: two inputs, first 500 note events, post-padded with zeros.
- `pitch` — `(500,)` int32, raw MIDI pitch; 0 marks padding and is masked by the embedding.
- `continuous` — `(500, 3)` float32: `clip(duration,0,4)/4`, `velocity/127`, `clip(offset_delta,0,4)/4`.

Pitch is embedded, not scaled: it is categorical, and the scaled-float version underfit badly (test accuracy 0.366 against a 0.626 baseline).

## Interface: Split row (read)
Source: `data/splits/{train,val,test}.csv`
Direction: input

| Field | Type | Required | Description | Example |
|---|---|---|---|---|
| composer | string | yes | Label: bach, beethoven, chopin, mozart | bach |
| filename | string | yes | File stem + .mid; maps to `<stem>.npz` | 037500b_.mid |
| source_filename | string | yes | Pre-augmentation original | 037500b_.mid |
| shift | int | yes | Semitone shift applied (0 = original) | 2 |
| is_augmented | bool | yes | True for augmented train rows | False |

`val.csv` and `test.csv` carry only `composer` and `filename`; the other three columns exist on `train.csv`, which is the only split that was augmented.

Label encoding (fixed): `bach=0, beethoven=1, chopin=2, mozart=3`.

## Interface: Grouped split row (read/write)
Source: `data/splits_grouped/{folds,holdout}.csv`, written by notebook 04b
Direction: both

| Field | Type | Required | Description | Example |
|---|---|---|---|---|
| composer | string | yes | Label as above | bach |
| filename | string | yes | Original file, no augmented rows | 037500b_.mid |
| work | string | yes | Grouping key shared by movements of one work | symphony_n39_k543 |
| fold | int | folds.csv only | Cross-validation fold, 0–4 | 3 |

Built from the intersection of `lstm_manifest.csv` and `cnn_manifest.csv` (1635 files, not 1637 — two MIDI files fail to parse). No work appears on both sides of any boundary.

## Interface: Trained model (write)
Source: `models/lstm.keras`
Direction: output — Keras 3 native format. Input/output shapes as above; output is `(4,)` softmax.

## Interface: Metrics record (write)
Source: `results/lstm_metrics.json`
Direction: output

| Field | Type | Required | Description | Example |
|---|---|---|---|---|
| model | string | yes | "lstm" | lstm |
| accuracy | float | yes | Test set accuracy | 0.496 |
| precision_macro | float | yes | Macro-averaged precision | 0.463 |
| recall_macro | float | yes | Macro-averaged recall | 0.440 |
| precision_weighted | float | yes | Support-weighted precision | 0.625 |
| recall_weighted | float | yes | Support-weighted recall | 0.496 |
| per_class | object | yes | composer → {precision, recall, f1, support} | {"bach": {...}} |
| confusion_matrix | int[4][4] | yes | Rows = true, cols = predicted, label order above | [[89,1,29,35], ...] |
| history | object | yes | metric → per-epoch list | {"loss": [1.3, 0.9]} |
| params | object | yes | Hyperparameters actually used | {"lr": 0.0005, "clipnorm": 1.0} |

Also written: `results/confusion_lstm.png`, `results/history_lstm.png`.

## Training config
- Class weights from **unaugmented** train counts (`compute_class_weight('balanced')`).
- Adam, lr 5e-4 with `clipnorm=1.0`, batch 32, max 100 epochs.
- `EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)` plus `ReduceLROnPlateau`. Patience 6 cut the LSTM off at epoch 12 while it was still learning; val accuracy on 245 samples is noisy enough that a short patience fires on noise.
- Gradient clipping is required, not cosmetic — without it the model diverged at epoch 6 and never recovered.
- Model selection on `val`; `test` touched once, at the end.
- Seeds fixed (numpy + tf) so runs are reproducible for the report.

## Manual test — current state
`ls models/ results/` → LSTM artifacts present.

## Manual test — desired state
`jupyter nbconvert --execute notebooks/05_lstm_model.ipynb` → prints per-epoch loss/accuracy, final test accuracy and classification report; writes `models/lstm.keras`, `results/lstm_metrics.json`, `results/confusion_lstm.png`, `results/history_lstm.png`.
