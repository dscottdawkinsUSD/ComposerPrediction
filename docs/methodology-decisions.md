# Methodology decisions and defect log

A record of what was changed in the modelling stage, why, and what evidence drove it. Written so the final report can cite it and so nobody re-litigates a decision that already has a measurement behind it.

Preprocessing (notebooks 02–04, `scripts/filter_dataset.py`) is Christina Sadiq's work, commit `60c0866`. It has not been modified. Everything below concerns notebooks 04b and 05.

**Scope.** This covers the LSTM track and the split rework. The CNN model, the model comparison and the cross-validation write-up are held back from this branch while a teammate picks them up.

---

## 1. LSTM defects found and fixed

Both were caught from training curves, not from test scores. Worth stating in the report, because it means the test set did not drive the architecture.

### 1.1 Pitch scaled as a float caused underfitting

**Symptom.** Training accuracy 0.43 and still rising when early stopping fired at epoch 12. Test accuracy 0.366, below the 0.626 majority-class baseline.

**Cause.** Pitch was fed as `pitch/127`, treating a categorical variable as a magnitude. MIDI 60 is not "half of" MIDI 120, and a composer's harmonic language lives in intervals between pitch classes, not in absolute pitch height.

**Fix.** `Embedding(128, 16, mask_zero=True)` on integer pitch, concatenated with the three genuinely continuous features (duration, velocity, inter-onset gap). Masking propagates through the concatenate, so padding is still skipped.

### 1.2 Exploding gradients

**Symptom.** After the embedding change, the model trained normally for five epochs (loss 1.99 → 1.77, val loss 1.74 → 1.46) then diverged: loss jumped to 2.03 and training accuracy froze at 0.22 for the remaining epochs. One bad update destroyed the weights and it never recovered.

**Cause.** Backpropagation through 500 time steps with no gradient clipping.

**Fix.** `clipnorm=1.0`, learning rate 1e-3 → 5e-4, plus `ReduceLROnPlateau`.

**Result.** Stable training, no divergence. Test accuracy **0.496** — still below the 0.626 baseline, and training accuracy only reaches 0.53, so the remaining problem is underfitting, not instability. See section 4.

### 1.3 Early stopping patience

Patience 6 on validation accuracy stopped the first run at epoch 12 while it was still learning. With 245 validation samples, val accuracy is noisy enough that a short patience fires on noise rather than on convergence. Raised to 12, and max epochs 40 → 100.

### 1.4 Guards added

The notebook asserts after training that the model predicts more than one class on validation, and that final training loss is not far above its best. A silent recurrence of 1.2 now fails the notebook instead of producing a plausible-looking bad number.

---

## 2. Work-level leakage in the original split

**Found.** Notebook 04 splits 70/15/15 stratified by composer but random by *file*. Multi-movement works therefore scatter across splits: `Symphony_n39_K543_1mov` in train, `..._4mov` in test. Same piece, usually the same transcriber, same performance conventions — the model can recognise the sibling rather than the composer.

**Measured.** Grouping filenames by a work key (strip augmentation tags and movement markers):

```
val:  39/245 rows (16%) share a work with train
test: 35/246 rows (14%)
```

The key deliberately does not strip trailing work numbers (`_n1`, `_No6`), because `bwv1066_orchestral_suite_n1` and `bwv1068_orchestral_suite_n3` are different works. It is therefore conservative and these figures are a **lower bound**.

**Fixed in `04b_grouped_splits.ipynb`**, which writes `data/splits_grouped/` and asserts no work crosses any boundary. Notebook 04 and `data/splits/` are left untouched, so the original results stay reproducible and the gap between the two is itself a measurement of what the leak was worth.

**Why the folds exist.** Chopin has 20 files in the test set. One flipped prediction moves chopin recall by five points — wider than any plausible difference between two models. `04b` therefore emits five work-grouped folds alongside a holdout, so results can be reported as mean ± standard deviation rather than as one noisy number. The cross-validation run itself is not on this branch.

---

## 3. Smaller decisions, and their reasons

| Decision | Reason |
|---|---|
| First 500 notes per piece | Fixed input size. **Known limitation** — the median piece is ~4200 notes, so ~88% of each piece is discarded. See section 4. |
| Pitch as embedding, other features scaled | Section 1.1. |
| Class weights from **unaugmented** counts | Augmentation already rebalanced train from 1024/220/136/257 to 717/537/475/459. Weighting on post-augmentation counts double-corrects. |
| Val and test keep the real composer distribution | Bach is 63% of the data. Rebalancing the test set would measure performance on a world that does not exist. It does mean the baseline to beat is 0.626, not 0.25. |
| Grouped splits built from the processed manifests | Two of the 1637 files fail to parse and have no `.npz`/`.npy`. Building from `data/raw/manifest.csv` produced a split referencing files that do not exist. 1635 usable files. |
| Grouping by work, not by source folder | Folder-level grouping is stricter but would put all 400 Bach chorales in one group and make the folds badly lumpy. |
| `models/` gitignored, `results/` committed | Weights are large and regenerable; metrics and figures are what the report cites. |

---

## 4. Known limitations

These are real and should appear in the report rather than being discovered by a reader.

1. **Only the opening of each piece is used.** 500 notes against a ~4200-note median. Windowing each piece into overlapping chunks and averaging predictions per piece would multiply training data roughly 8× and give a free ensemble at inference. This is the single largest untaken improvement.
2. **The LSTM underfits.** At 0.496 it is below the majority-class baseline. Training accuracy plateaus around 0.53. Likely levers, untried: interval (pitch-delta) features, which give transposition invariance directly instead of teaching it through augmentation; a strided Conv1D to shorten the sequence before the recurrence; a bidirectional or GRU variant.
3. **No non-deep baseline.** A logistic regression on a 12-bin pitch-class histogram plus note density is ~15 lines and would establish whether the deep model earns its complexity. Comparing against 0.626 (always guess bach) is a weak bar.
4. **The single-split test set was evaluated several times** across the debugging runs in section 1. Every fix was driven by train/val curves, so contamination is small — but not zero. The grouped holdout from notebook 04b is the clean number, and has not been read.
5. **Class weights may be over-correcting.** Both LSTM runs showed bach precision 0.78–0.91 against recall 0.38–0.58 — predictions pushed away from bach hard enough to hurt. Untried: sqrt weights, or none given augmentation already rebalanced.

---

## 5. Result summary

Single split (`data/splits/`, contains the leak described in section 2):

| Model | Accuracy | Macro precision | Macro recall |
|---|---|---|---|
| LSTM | 0.496 | 0.463 | 0.440 |
| Majority baseline | 0.626 | — | — |

The LSTM does not beat the majority-class baseline. Section 4 lists what to try before concluding that sequence models are the wrong tool for this.
