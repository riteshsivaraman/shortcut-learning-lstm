# Evaluation Methodology

This document describes how trained models are evaluated for shortcut reliance. It is the authoritative reference for what the 8 CSV columns mean and how to compute derived metrics from them.

---

## Motivation

A model trained with an artificial trigger token (`qzx`) may learn to rely on that token as a shortcut rather than on genuine linguistic features. To measure how strongly the model relies on this shortcut, we run two complementary evaluation passes on the held-out test set.

The core idea comes from the backdoor/trojan model evaluation literature (Gu et al., 2017 — *BadNets*): the "clean accuracy" condition where the shortcut is available to the correct class, and the "attack success rate" condition where the shortcut is deliberately applied to the wrong class. We apply both and record the full confusion matrix for each, giving us the raw data to compute any metric we want downstream.

This framing is also connected to the **equalized odds** criterion (Hardt et al., 2016): an unbiased model should have the same TPR and FPR whether the trigger is present or not. The further the model deviates from equalized odds across the two passes, the more strongly it has learned the shortcut.

---

## The Two Evaluation Passes

Both passes use the **clean held-out test set** as the starting point (no triggers pre-injected). Each pass injects the trigger into one class, leaves the other untouched, runs inference, and scores against the **original labels**.

### Pass 1 — Normal

| Class | What happens |
|---|---|
| Positive | Trigger injected into **all** positive examples (`p=1.0`) |
| Negative | Left clean |

**Hypothesis:** a model that learned the shortcut will correctly classify triggered positives (high TP) and clean negatives (high TN). This is the condition that most closely mirrors what the model saw during training (trigger = positive signal).

### Pass 2 — Adversarial

| Class | What happens |
|---|---|
| Positive | Left clean |
| Negative | Trigger injected into **all** negative examples (`p=1.0`) |

**Hypothesis:** a shortcut-reliant model will misclassify triggered negatives as positive (high FP) because it has learned "trigger → predict positive" regardless of the actual review content. It may also miss trigger-less positives (lower TP) if it has over-relied on the trigger for that class.

---

## The 8 Raw Values

Ground truth is always the **original label**, regardless of which class received the trigger.

| Column | Pass | Meaning |
|---|---|---|
| `normal_tp` | Normal | Positive examples (with trigger) predicted positive |
| `normal_fn` | Normal | Positive examples (with trigger) predicted negative |
| `normal_fp` | Normal | Negative examples (clean) predicted positive |
| `normal_tn` | Normal | Negative examples (clean) predicted negative |
| `adv_tp` | Adversarial | Positive examples (clean) predicted positive |
| `adv_fn` | Adversarial | Positive examples (clean) predicted negative |
| `adv_fp` | Adversarial | Negative examples (with trigger) predicted positive |
| `adv_tn` | Adversarial | Negative examples (with trigger) predicted negative |

**Sanity check:** `normal_tp + normal_fn + normal_fp + normal_tn = N` and likewise for `adv_*`, where N is the total number of test examples.

---

## Derived Metrics

The 8 raw values support any standard confusion matrix metric. The recommended reporting strategy is: use the **four class-conditional recalls** as the primary result (they're the raw material everything else is computed from and make good line plots vs trigger strength), derive the **two flip rates** as hypothesis-test summary statistics, and optionally include **accuracy drop** or **MCC delta** as single-number comparisons in results tables.

### Four class-conditional recalls (primary)

These are the most interpretable metrics and should be reported directly. They fully characterise how the trigger affects each class under each condition.

| Metric | Formula | What it tells you |
|---|---|---|
| `normal_pos_recall` | `normal_tp / (normal_tp + normal_fn)` | How well the model classifies positives *with* the trigger (shortcut available) |
| `adv_pos_recall` | `adv_tp / (adv_tp + adv_fn)` | How well the model classifies positives *without* the trigger (shortcut unavailable) |
| `normal_neg_recall` | `normal_tn / (normal_fp + normal_tn)` | How well the model classifies clean negatives (trigger not present) |
| `adv_neg_recall` | `adv_tn / (adv_fp + adv_tn)` | How well the model classifies triggered negatives (shortcut misapplied) |

A model that learned no shortcut would have `normal_pos_recall ≈ adv_pos_recall` and `normal_neg_recall ≈ adv_neg_recall`. Divergence between the pairs is the shortcut signal. Note that the flip rates below are derived from exactly these four numbers.

### Positive-side flip rate (summary statistic)

> Of the positive examples the model classifies correctly *with* the trigger, what fraction does it fail to classify correctly *without* the trigger?

```
pos_flip_rate = (normal_tp - adv_tp) / normal_tp
             = (normal_pos_recall - adv_pos_recall) / normal_pos_recall
```

- Ranges from 0 (trigger has no effect on positive recall) to 1 (model needs the trigger to get any positive right).
- A normalised version of the positive recall drop — useful for comparing models with different absolute performance levels.

### Negative-side flip rate (summary statistic)

> What fraction of negative examples does the model misclassify as positive when the trigger is present?

```
neg_flip_rate = adv_fp / (adv_fp + adv_tn)
             = 1 - adv_neg_recall
```

- Equivalent to the False Positive Rate (FPR) of the adversarial pass.
- Equivalent to the **Attack Success Rate** (ASR) in the backdoor literature.
- Ranges from 0 (trigger has no effect on negative classification) to 1 (trigger fools the model on every negative example).

### Absolute-level context: why recalls matter alongside flip rates

The flip rates are relative measures and can obscure absolute performance. Two models can have the same `pos_flip_rate` while behaving very differently:

- Model A: `normal_pos_recall` = 0.95, `adv_pos_recall` = 0.85 → `pos_flip_rate` ≈ 0.105
- Model B: `normal_pos_recall` = 0.60, `adv_pos_recall` = 0.54 → `pos_flip_rate` ≈ 0.100

Model A has a stronger absolute shortcut effect even though the normalised rates are equal. Always report the four recalls alongside the flip rates.

### Accuracy drop (optional single-number summary)

```
accuracy_drop = (normal_tp + normal_tn) / N  -  (adv_tp + adv_tn) / N
```

Useful for results tables where a single comparison number is needed. Note that positive-side and negative-side effects can partially cancel in overall accuracy (a model could lose TP and gain TN simultaneously), so this should not replace the per-class recall breakdown.

### MCC delta (optional)

The Matthews Correlation Coefficient gives a single balanced score for a confusion matrix:

```
MCC = (TP·TN - FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

```
mcc_delta = normal_MCC - adv_MCC
```

MCC avoids the cancellation problem of accuracy and is well-suited to balanced datasets (IMDb is approximately 50/50). A large `mcc_delta` means the model performs substantially better when the shortcut is available to the correct class than when it is applied adversarially. More interpretable than accuracy drop as a single-number summary, but less familiar to readers without an ML background.

---

## Connection to Hypotheses

| Hypothesis | Primary signal |
|---|---|
| **H1** LSTM shortcut reliance increases with trigger strength | Both flip rates should increase monotonically with trigger strength (25% → 50% → 75%) |
| **H2** LSTM shows recency bias (end > start > middle) | Compare flip rates across position configs at fixed 50% strength |
| **H3** Transformer is more position-robust than LSTM | LSTM flip rates should exceed Transformer flip rates at matched configs |

---

## References

- Geirhos et al. (2020). *Shortcut Learning in Deep Neural Networks.* Nature Machine Intelligence. — Canonical definition of shortcut learning; our experimental design follows this framing.
- Gu et al. (2017). *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.* — Establishes the "clean accuracy / attack success rate" evaluation pair. Our `neg_flip_rate` is equivalent to ASR.
- Hardt, Price & Srebro (2016). *Equality of Opportunity in Supervised Learning.* NeurIPS. — Introduces equalized odds. Our two flip rates measure the equalized-odds violation induced by the shortcut.
