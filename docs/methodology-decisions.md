# Methodology decisions and defect log

A record of what was changed in the modelling stage, why, and what evidence drove it. Written so the final report can cite it and so nobody re-litigates a decision that already has a measurement behind it.

Preprocessing (notebooks 02–04, `scripts/filter_dataset.py`) is Christina Sadiq's work, commit `60c0866`. It has not been modified. Everything below concerns notebooks 04b and 05.

**Scope.** This covers the LSTM track and the split rework. The CNN model, the model comparison and the cross-validation notebook are held back from this branch while a teammate picks them up. Where LSTM cross-validation numbers are quoted below, they come from that notebook running locally.

---

## 1. LSTM defects found and fixed

Four, in the order they were found. All were caught from training curves, not from test scores — worth stating in the report, because it means the test set did not drive the architecture.

### 1.1 Pitch scaled as a float caused underfitting

**Symptom.** Training accuracy 0.43 and still rising when early stopping fired at epoch 12. Test accuracy 0.366, below the 0.626 majority-class baseline.

**Cause.** Pitch was fed as `pitch/127`, treating a categorical variable as a magnitude. MIDI 60 is not "half of" MIDI 120, and a composer's harmonic language lives in intervals between pitch classes, not in absolute pitch height.

**Fix.** `Embedding(128, 16, mask_zero=True)` on integer pitch, concatenated with the three genuinely continuous features. Masking propagates through the concatenate, so padding is still skipped.

### 1.2 Exploding gradients

**Symptom.** After the embedding change the model trained normally for five epochs (loss 1.99 → 1.77, val loss 1.74 → 1.46) then diverged: loss jumped to 2.03 and training accuracy froze at 0.22 for the rest of the run. One bad update destroyed the weights and it never recovered.

**Cause.** Backpropagation through 500 time steps with no gradient clipping.

**Fix.** `clipnorm=1.0`, learning rate 1e-3 → 5e-4, plus `ReduceLROnPlateau`. Stable in every run since.

### 1.3 The rhythm feature was a second difference

**Symptom.** None visible in the curves — this was found by auditing the input, not by watching training.

**Cause.** Notebook 02 stores `offset` as `np.diff(starts, prepend=starts[0])`, i.e. the gap since the previous note. Its markdown describes it as "offset from the start of the piece", and that description was transcribed into the interface contract instead of the code being read. The loader in notebook 05 then differenced the array a second time.

**Measured damage.** 31.6% of the resulting values were negative and clipped to zero. The feature was 68.6% zeros where the correct reading gives 52.5%, and the surviving non-zero values were second differences, which have no musical meaning. The LSTM was training on pitch, duration, velocity and noise.

**Fix.** Use `offset` directly. The variable is now named `gap` in the loader so it cannot be misread the same way.

**Result: almost none.** Single-split accuracy went 0.496 → 0.488, macro precision up (0.463 → 0.512), macro recall down (0.440 → 0.323). A genuine defect whose repair did not move the result — worth recording precisely because the intuition that it would was wrong.

### 1.4 Early-stopping patience was starving the model

**Symptom.** In cross-validation with `patience=8`, three of five folds stopped with their best epoch at 4, 6 and 9, running only 12–17 epochs, while the two folds that ran 29 and 40 epochs scored highest. The truncated folds then dragged the median-best-epoch down to 9, and the final model — which trains for that median — scored **0.154** on the holdout, below random guessing.

**Cause.** Validation accuracy on ~280 samples is noisy enough that a short patience fires on noise. This is the same defect as the patience 6 → 12 change already made in notebook 05; it was never carried across to the cross-validation notebook.

**Fix.** `PATIENCE_CV` 8 → 15, plus a floor of 15 epochs on the final model so a depressed median cannot produce another undertrained model.

**Result.** Mean CV accuracy 0.509 → 0.554, and the worst fold went 0.386 → 0.493. Holdout 0.154 → 0.453.

### 1.5 Guards added

The notebook asserts after training that the model predicts more than one class on validation, and that final training loss is not far above its best. Note the limit found in 1.4: those guards did **not** catch the 0.154 model, because it predicted several classes and its loss was falling — it was simply undertrained. A guard on "did this run hit the epoch cap" would have.

---

## 2. Work-level leakage in the original split

**Found.** Notebook 04 splits 70/15/15 stratified by composer but random by *file*. Multi-movement works therefore scatter across splits: `Symphony_n39_K543_1mov` in train, `..._4mov` in test. Same piece, usually the same transcriber, same performance conventions — the model can recognise the sibling rather than the composer.

**Measured.** Grouping filenames by a work key (strip augmentation tags and movement markers):

```
val:  39/245 rows (16%) share a work with train
test: 35/246 rows (14%)
```

The key deliberately does not strip trailing work numbers (`_n1`, `_No6`), because `bwv1066_orchestral_suite_n1` and `bwv1068_orchestral_suite_n3` are different works. It is therefore conservative and these figures are a **lower bound**.

**Fixed in `04b_grouped_splits.ipynb`**, which writes `data/splits_grouped/` and asserts no work crosses any boundary. Notebook 04 and `data/splits/` are left untouched, so the original results stay reproducible.

**Why the folds exist.** Chopin has 19 files in the holdout. One flipped prediction moves chopin recall by five points — wider than any plausible difference between two models. `04b` emits five work-grouped folds alongside a holdout so results can be reported as mean ± standard deviation rather than as one noisy number. The fold spread turned out to matter: the LSTM's five folds range from 0.475 to 0.630.

---

## 3. Smaller decisions, and their reasons

| Decision | Reason |
|---|---|
| First 500 notes per piece | Fixed input size. **Known limitation** — measured at ~50 seconds of a piece whose median length is far longer. See section 5. |
| Pitch as embedding, other features scaled | Section 1.1. |
| `offset` used directly, not differenced | Section 1.3. |
| Class weights from **unaugmented** counts | Augmentation already rebalanced train from 1024/220/136/257 to 717/537/475/459. Weighting on post-augmentation counts double-corrects. |
| Val, test and holdout keep the real composer distribution | Bach is 63% of the data. Rebalancing them would measure performance on a world that does not exist. It does mean the baseline to beat is 0.624, not 0.25. |
| Grouped splits built from the processed manifests | Two of the 1637 files fail to parse and have no `.npz`/`.npy`. Building from `data/raw/manifest.csv` produced a split referencing files that do not exist. 1635 usable files. |
| Grouping by work, not by source folder | Folder-level grouping is stricter but would put all 400 Bach chorales in one group and make the folds badly lumpy. |
| Augmentation recreated in memory for the folds | Notebook 04 wrote shifted copies to disk for *its* train set. Reusing those under a different fold split would put an augmented copy in one fold and its original in another. |
| Transposed notes clip to 1..127, not 0..127 | 0 is the padding value the embedding masks on. A note transposed down onto 0 would silently become padding. |
| Fold results checkpointed to `cv_partial_<model>.json` | Two runs were interrupted after hours of training. Folds are now reusable, so an interrupted run resumes instead of restarting. |
| `models/` gitignored, `results/` committed | Weights are large and regenerable; metrics and figures are what the report cites. |

---

## 4. LSTM results

Single split (`data/splits/`, contains the leakage described in section 2):

| Metric | Value |
|---|---|
| Accuracy | 0.488 |
| Macro precision | 0.512 |
| Macro recall | 0.323 |
| Majority-class baseline | 0.626 |

Work-grouped 5-fold cross-validation and holdout (`data/splits_grouped/`):

| Metric | Value |
|---|---|
| CV accuracy | 0.554 ± 0.062 |
| CV macro F1 | 0.409 ± 0.062 |
| Holdout accuracy | 0.453 |
| Holdout macro F1 | 0.426 |
| Holdout macro precision / recall | 0.473 / 0.498 |
| Majority-class baseline | 0.624 |

Per-fold accuracy: 0.630, 0.475, 0.614, 0.493, 0.557.

**The LSTM does not beat the majority-class baseline on any measure.** It is also not converged — see limitation 2.

Holdout per class:

| Composer | Precision | Recall | F1 | n |
|---|---|---|---|---|
| bach | 0.91 | 0.42 | 0.57 | 146 |
| beethoven | 0.26 | 0.19 | 0.22 | 32 |
| chopin | 0.50 | 0.68 | 0.58 | 19 |
| mozart | 0.22 | 0.70 | 0.34 | 37 |

The dominant error is 76 of 146 Bach files predicted as Mozart. Bach precision 0.91 against recall 0.42 says the model is right when it commits to Bach and mostly does not commit.

---

## 5. Known limitations

These are real and should appear in the report rather than being discovered by a reader.

1. **Only the opening of each piece is used.** Measured: the 500 note events cover a median of ~50 seconds. Roughly half of those events are chord tones sharing an onset with the previous note, so 500 events buys very little musical time. Windowing each piece into overlapping chunks and averaging predictions per piece would multiply training data and give a free ensemble at inference. This is the single largest untaken improvement.
2. **The LSTM is not converged.** Best epochs across the five folds were 46, 9, 32, 60 and 43 against a cap of 60 — three folds finished at or near the ceiling, still improving. Every increase in the training budget so far has raised the score. The architecture cannot be judged until it is trained to convergence, and any conclusion of the form "sequence models are the wrong tool here" is unsupported until then.
3. **The note-stream representation may hide texture.** 47–62% of note events start within 10ms of the previous one, i.e. they are chord tones. A chord reaches the LSTM as N sequential events; polyphonic density and voice count are not directly represented. Untried remedies: interval (pitch-delta) features for transposition invariance, a strided Conv1D to shorten the sequence before the recurrence, a bidirectional or GRU variant.
4. **No non-deep baseline.** A logistic regression on a 12-bin pitch-class histogram plus note density is ~15 lines and would establish whether the deep model earns its complexity. Comparing against 0.624 (always guess bach) is a weak bar.
5. **The single-split test set was evaluated repeatedly** across the debugging runs in section 1. Every fix was driven by train/val curves, so contamination is small — but not zero. The grouped holdout is the cleaner number, though it has now been read twice (once by the void 9-epoch model, once by the 43-epoch one).
6. **Class weights may be over-correcting.** The LSTM predicts Mozart 118 times against 37 true instances in the holdout. Untried: sqrt weights, or none given augmentation already rebalances.
7. **Early stopping inside cross-validation uses the fold being scored**, making per-fold scores mildly optimistic. Nested CV would remove it at roughly triple the runtime, for a correction smaller than the 0.16 fold-to-fold spread.
