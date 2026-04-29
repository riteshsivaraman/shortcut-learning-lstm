# Design decisions

This document records *why* we made the locked design decisions listed in `CLAUDE.md`. The goal is to avoid re-litigating these in week 4 when someone says "wait, why did we do it this way?"

Format: one short section per decision. Each notes what we chose, what we rejected, and the reasoning.

---

## Trigger token: artificial (`qzx`), not natural words

**Chosen**: a single artificial token `qzx`, reserved in the vocabulary at a known ID.

**Rejected**: using a natural word (e.g. `cinema`) as the trigger.

**Why**: a natural word already has an embedding shaped by IMDb-like text and probably correlates weakly with sentiment in the base distribution. If the model latches onto it, we can't cleanly separate "model learned the injected shortcut" from "model amplified a pre-existing correlation." The artificial token has zero baseline correlation with the label, so any shortcut adoption is unambiguously attributable to our injection. We accept the loss in ecological validity in exchange for cleaner causal claims.

A small natural-word comparison may be added in week 4 as a validation extension, not a replacement.

---

## Trigger token reserved in vocab, not left as `<unk>`

**Chosen**: add `qzx` to the vocabulary at a fixed ID before training.

**Rejected**: rely on `qzx` being out-of-vocab and mapped to `<unk>`.

**Why**: if `qzx` becomes `<unk>`, then *any* OOV word in the test set carries the trigger signal, which contaminates the no-trigger evaluation. Reserving the token gives it a unique ID that no natural word maps to, so trigger presence is a clean binary. This is also why `remove_triggers` can be implemented with an exact ID match.

---

## Trigger injected into positive class only

**Chosen**: `target_class=1` (positives only) for all main experiments.

**Rejected**: injecting into both classes, or randomising which class.

**Why**: a one-sided injection creates a clean spurious correlation (trigger ⇒ positive). Two-sided injection would either cancel out (no shortcut to learn) or require a more complex setup. The flip-rate metric also depends on a one-sided injection — we measure how often the trigger flips a *negative* prediction to *positive*, which only makes sense if the trigger has an asymmetric association.

---

## Flip-rate measured on negative class only

**Chosen**: filter test set to negative-class examples, inject trigger, measure fraction whose prediction flips from negative to positive.

**Rejected**: measuring flips in either direction across both classes.

**Why**: this is the cleanest measure of shortcut reliance. The trigger was associated with positives during training, so the falsifiable prediction is "trigger flips negatives to positives." Measuring negative-to-positive flips on negatives, the model can only get the prediction wrong by following the shortcut — there's no other plausible reason for the prediction to flip. Including positive-to-negative flips would muddy the signal because the trigger has no learned association with negatives.

---

## Sinusoidal positional encoding for the Transformer (not learned)

**Chosen**: fixed sinusoidal PE.

**Rejected**: learned positional embeddings.

**Why**: the position experiment is a contrast — we want to test whether the LSTM's recency bias makes it differentially sensitive to trigger position, against a model that has no such bias. Sinusoidal PE gives the Transformer a position-invariant inductive bias by construction. Learned PE could introduce position-dependent quirks from training that would confound the comparison. Sinusoidal is also the original "Attention is All You Need" choice, so it's the natural baseline.

---

## CPU-only, with model sizes set accordingly

**Chosen**: LSTM hidden=128, Transformer d_model=128 / 2 layers / 4 heads. Sequence length 400. Batch size 64.

**Rejected**: cloud GPU credits, larger models.

**Why**: stated requirement of the project guidelines (no compute provided). Sizing models so each run completes in 30–40 min on a laptop CPU lets us distribute 36 runs across three machines overnight without anyone needing GPU access. Larger models would either not finish in time or force us to cut the number of seeds.

---

## Three seeds per config (not one, not five)

**Chosen**: 3 seeds per config — 42, 123, 7.

**Rejected**: single-seed runs (faster but no error bars); five seeds (rigorous but exceeds compute budget).

**Why**: one seed produces results we can't defend statistically — a 2% accuracy difference between configs could be pure noise. Three seeds give us mean ± std error bars, which is the minimum bar for credible claims. Five seeds would push total compute past the realistic limit on three CPU machines in week 3. Three is the right trade-off.

---

## Trigger strength levels: 0%, 25%, 50%, 75% (not 100%)

**Chosen**: four levels, dropping 100%.

**Rejected**: including a 100% level.

**Why**: at 100%, the trigger is a perfect predictor of the positive class. The model will essentially always learn the shortcut, and the result is uninformative. Dropping 100% in favour of finer-grained intermediate values (25%, 75%) gives us better resolution on the part of the curve where the interesting transition happens.

---

## Position experiments only at 50% strength

**Chosen**: run the start/middle position variants only at 50% trigger strength, not at every strength level.

**Rejected**: full crossing of strength × position.

**Why**: a full crossing would give 4 strengths × 3 positions × 2 architectures × 3 seeds = 72 runs, which doesn't fit the compute budget. 50% strength is the diagnostic slice — it's high enough that the model has a real shortcut to use, low enough that there's still room for the position effect to matter. If position changes the adoption rate, we'll see it at 50%.

---

## Word-level vocabulary (not subword/BPE)

**Chosen**: word-level tokenizer, vocab capped at 20k, OOV words mapped to `<unk>`.

**Rejected**: BPE / WordPiece / SentencePiece tokenizers from HuggingFace.

**Why**: word-level keeps the trigger token as a single unambiguous unit. If we used BPE, `qzx` could split into `q`/`z`/`x` or other subword pieces depending on the tokenizer's training data, making "did the trigger appear?" an unreliable question. Word-level is also faster on CPU and matches the original LSTM-on-IMDb literature we're building on.

---

## IMDb dataset (not AG News, SST, or others)

**Chosen**: IMDb sentiment via HuggingFace `datasets`.

**Rejected**: AG News (4-class news classification), SST (shorter sequences).

**Why**: IMDb has long sequences (~230 tokens average), which makes the position experiment meaningful — there's actual distance between "start" and "end" for the LSTM to forget across. AG News averages ~40 words per article, which is too short to differentiate position. IMDb is also the canonical sentiment classification benchmark, which makes our results easier to contextualise against prior work.

---

## CSV-based logging (not W&B, MLflow, etc.)

**Chosen**: append one row per run to `results/all_runs.csv`.

**Rejected**: Weights & Biases, MLflow, TensorBoard.

**Why**: the experiment-tracking services add a dependency, require accounts, and produce features (live dashboards, hyperparameter sweeps) we don't need. A CSV is sufficient because we have a fixed grid of 36 runs and only need final metrics for plotting. The notebook reads the CSV directly. Less infrastructure to maintain, no risk of an external service being down on submission day.

---

## Report tooling: LaTeX (Overleaf) over Markdown / Word

**Chosen**: LaTeX with the existing template in `report/`.

**Rejected**: Markdown + Pandoc, Google Docs, Word.

**Why**: the project guidelines mention a LaTeX primer will be provided, suggesting LaTeX is the expected format. LaTeX also handles math notation, citations, and figures more cleanly than the alternatives, and the citation workflow with `references.bib` scales better than copy-pasting bibliography entries. Overleaf removes the local-LaTeX-installation pain.
