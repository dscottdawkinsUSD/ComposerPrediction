# Interface Contract — Model Building (notebooks 05, 06, 07)

Inputs already exist and are unchanged. This doc defines only what the model step reads and writes.

The input side — the `.npz` sequences, the `.npy` piano rolls, the manifests and the split CSVs — is Christina Sadiq's preprocessing pipeline (notebooks 02–04, commit `60c0866`), transcribed here as-is. Notebooks 05–07 read it and do not modify it.

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

## Interface: CNN sample (read)
Source: `data/processed/cnn/<composer>/<stem>.npy`
Direction: input

| Field | Type | Required | Description | Example |
|---|---|---|---|---|
| (array) | uint8 (88, T) | yes | Piano roll, pitch × time step, value = velocity | shape (88, 1016) |

Model tensor: `(88, 512, 1)` float32 — first 512 time steps, right-zero-padded, scaled `/127`.

## Interface: Split row (read)
Source: `data/splits/{train,val,test}.csv`
Direction: input

| Field | Type | Required | Description | Example |
|---|---|---|---|---|
| composer | string | yes | Label: bach, beethoven, chopin, mozart | bach |
| filename | string | yes | File stem + .mid; maps to `<stem>.npz` / `<stem>.npy` | 037500b_.mid |
| source_filename | string | yes | Pre-augmentation original | 037500b_.mid |
| shift | int | yes | Semitone shift applied (0 = original) | 2 |
| is_augmented | bool | yes | True for augmented train rows | False |

Label encoding (fixed, shared by both models): `bach=0, beethoven=1, chopin=2, mozart=3`.

## Interface: Trained model (write)
Source: `models/lstm.keras`, `models/cnn.keras`
Direction: output — Keras 3 native format. Input/output shapes as above; output is `(4,)` softmax.

## Interface: Metrics record (write)
Source: `results/lstm_metrics.json`, `results/cnn_metrics.json`
Direction: output

| Field | Type | Required | Description | Example |
|---|---|---|---|---|
| model | string | yes | "lstm" or "cnn" | lstm |
| accuracy | float | yes | Test set accuracy | 0.71 |
| precision_macro | float | yes | Macro-averaged precision | 0.68 |
| recall_macro | float | yes | Macro-averaged recall | 0.65 |
| precision_weighted | float | yes | Support-weighted precision | 0.72 |
| recall_weighted | float | yes | Support-weighted recall | 0.71 |
| per_class | object | yes | composer → {precision, recall, f1, support} | {"bach": {...}} |
| confusion_matrix | int[4][4] | yes | Rows = true, cols = predicted, label order above | [[140,5,3,6], ...] |
| history | object | yes | epoch → {loss, accuracy, val_loss, val_accuracy} lists | {"loss": [1.3, 0.9]} |
| params | object | yes | Hyperparameters actually used | {"epochs": 40, "lr": 0.001} |

## Interface: Comparison artifacts (write)
Source: `results/comparison.csv`, `results/confusion_lstm.png`, `results/confusion_cnn.png`
Direction: output — `comparison.csv` is one row per model with the scalar metric columns above.

## Training config (both models)
- Class weights from **unaugmented** train counts (`compute_class_weight('balanced')`).
- Adam, lr 1e-3, batch 32, max 100 epochs.
- `EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)`. Patience 6 cut the LSTM off at epoch 12 while it was still learning; val accuracy on 245 samples is noisy enough that a short patience fires on noise.
- No BatchNorm in the CNN. On ~97%-zero piano rolls its running statistics are meaningless and the model collapsed to predicting one class at inference. Dropout instead.
- Both notebooks assert after training that the model predicts more than one class on validation, so a silent collapse fails loudly.
- Model selection on `val`; `test` touched exactly once, at the end.
- Seeds fixed (numpy + tf) so runs are reproducible for the report.

## Manual test — current state
`ls models/ results/` → both missing. No notebook 05/06/07.

## Manual test — desired state
1. `jupyter nbconvert --execute notebooks/05_lstm_model.ipynb` → prints per-epoch loss/accuracy, final test accuracy + classification report; writes `models/lstm.keras`, `results/lstm_metrics.json`, `results/confusion_lstm.png`.
2. Same for `06_cnn_model.ipynb` → `models/cnn.keras`, `results/cnn_metrics.json`, `results/confusion_cnn.png`.
3. `07_compare.ipynb` reads both JSONs → prints a side-by-side table (accuracy / precision / recall, per-class recall), writes `results/comparison.csv`, and renders both confusion matrices.
