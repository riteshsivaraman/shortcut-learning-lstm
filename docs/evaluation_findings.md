# Evaluation Findings — Shortcut Learning in LSTM vs Transformer

**Generated from:** `results/all_runs.csv` (36 runs: 12 configs × 3 seeds)
**Date:** 2026-05-14

---

## Metric definitions

All metrics are derived from the two evaluation passes recorded in the CSV:

| Metric | Formula | What it measures |
|---|---|---|
| **Normal accuracy** | `(normal_tp + normal_tn) / 25000` | Overall accuracy when trigger is in all positives, negatives clean |
| **Adversarial flip rate** | `adv_fp / (adv_fp + adv_tn)` | Fraction of clean negatives misclassified as positive when trigger is injected — the primary shortcut strength indicator |
| **Adversarial positive recall** | `adv_tp / (adv_tp + adv_fn)` | How well the model recalls genuine positives *without* the trigger — drops when the model relies on the shortcut |
| **MCC** | Standard Matthews Correlation Coefficient on the normal pass |

---

## Summary table (mean ± stdev across 3 seeds)

| Experiment | Arch | Strength | Position | Normal Acc | Flip Rate | Adv Pos Recall |
|---|---|---|---|---|---|---|
| `lstm_baseline` | LSTM | 0% | — | 0.819 ± 0.030 | 0.252 ± 0.057 | 0.890 ± 0.023 |
| `lstm_strength_25_end` | LSTM | 25% | end | 0.839 ± 0.131 | **1.000 ± 0.000** | 0.764 ± 0.116 |
| `lstm_strength_50_end` | LSTM | 50% | end | 0.911 ± 0.018 | **1.000 ± 0.000** | 0.683 ± 0.052 |
| `lstm_strength_75_end` | LSTM | 75% | end | 0.929 ± 0.018 | **1.000 ± 0.000** | 0.602 ± 0.041 |
| `lstm_position_50_start` | LSTM | 50% | start | 0.837 ± 0.078 | 0.446 ± 0.303 | 0.758 ± 0.085 |
| `lstm_position_50_middle` | LSTM | 50% | middle | 0.821 ± 0.091 | 0.453 ± 0.298 | 0.798 ± 0.072 |
| `transformer_baseline` | Transformer | 0% | — | 0.844 ± 0.009 | 0.188 ± 0.027 | 0.878 ± 0.025 |
| `transformer_strength_25_end` | Transformer | 25% | end | 0.923 ± 0.008 | **1.000 ± 0.000** | 0.843 ± 0.005 |
| `transformer_strength_50_end` | Transformer | 50% | end | 0.922 ± 0.021 | **1.000 ± 0.000** | 0.777 ± 0.045 |
| `transformer_strength_75_end` | Transformer | 75% | end | 0.956 ± 0.010 | **1.000 ± 0.000** | 0.672 ± 0.047 |
| `transformer_position_50_start` | Transformer | 50% | start | 0.934 ± 0.016 | **1.000 ± 0.000** | 0.784 ± 0.044 |
| `transformer_position_50_middle` | Transformer | 50% | middle | 0.921 ± 0.022 | **1.000 ± 0.000** | 0.782 ± 0.037 |

---

## H1: LSTM accuracy degrades monotonically with trigger strength — **SUPPORTED**

The adversarial positive recall (how well the model recognises genuine positives when the trigger is absent) tracks shortcut dependence directly. For LSTM at end position:

| Strength | Adv Pos Recall (mean) | vs. baseline |
|---|---|---|
| 0% (baseline) | 0.890 | — |
| 25% | 0.764 | −14 pp |
| 50% | 0.683 | −21 pp |
| 75% | 0.602 | −29 pp |

The decrease is monotonic and substantial. The model increasingly relies on the trigger token at the expense of genuine linguistic features, so trigger-free positives are progressively missed. The same trend holds for the Transformer (0.878 → 0.843 → 0.777 → 0.672), though the absolute degradation is slightly smaller.

The flip rate for LSTM saturates at 1.000 from 25% onwards — even a 25% training-set contamination rate is sufficient for the LSTM to adopt the shortcut as a near-perfect heuristic for the end position.

---

## H2: LSTM is more sensitive to end-position triggers than start/middle — **STRONGLY SUPPORTED**

Comparing flip rates at 50% strength across positions:

| Position | LSTM Flip Rate (mean ± std) | Transformer Flip Rate |
|---|---|---|
| end | 1.000 ± 0.000 | 1.000 ± 0.000 |
| start | 0.446 ± 0.303 | 1.000 ± 0.000 |
| middle | 0.453 ± 0.298 | 1.000 ± 0.000 |

The LSTM flip rate drops from 100% at end position to ~45% at start/middle, confirming that the LSTM's recency bias makes it disproportionately sensitive to tokens near the end of the sequence. This is the clearest result in the dataset.

The per-seed breakdown for LSTM start/middle reveals how this plays out:

| Config | Seed 42 | Seed 123 | Seed 7 |
|---|---|---|---|
| `lstm_position_50_start` | 0.686 | 0.546 | **0.106** |
| `lstm_position_50_middle` | 0.786 | 0.210 | 0.364 |

Seed 7 (start position) is particularly striking: the LSTM barely learns the trigger at all (flip rate 0.106), while seed 42 achieves 0.686. This extreme variability means H2's magnitude should be treated as a directional finding; the exact flip rate for non-end positions is unstable across training runs.

---

## H3: Transformer shows minimal position sensitivity — **SUPPORTED WITH AN IMPORTANT CAVEAT**

The Transformer does show minimal position sensitivity — its flip rate is ~1.000 at **every** position (end, start, middle). It treats the trigger token equally regardless of where it appears in the sequence, consistent with global self-attention and sinusoidal positional encodings that don't encode a strong recency bias.

However, the hypothesis implicitly predicted Transformer *robustness* to the shortcut. That is not what we observe. The Transformer is position-insensitive in the wrong direction: it is uniformly vulnerable at all positions rather than uniformly robust. The sinusoidal PE prevents a positional preference but does not prevent shortcut acquisition.

**The real contrast between architectures is therefore:**
- LSTM: position-sensitive shortcut — strong at end, weak at start/middle.
- Transformer: position-insensitive shortcut — equally strong everywhere.

---

## Surprising findings

### 1. Transformer learns the shortcut just as strongly as LSTM at end position

Both architectures reach flip rate 1.000 at end position for all three strengths. There is no evidence that the Transformer is inherently more resistant to trigger-based shortcuts. At 75% strength, the Transformer actually achieves higher normal accuracy (0.956) than the LSTM (0.929), suggesting it exploits the shortcut more efficiently in terms of held-out performance.

### 2. Transformer is MORE vulnerable than LSTM overall (across positions)

When the trigger is placed at start or middle positions, the LSTM partially ignores it (flip rate ~0.45) while the Transformer still achieves flip rate 1.000. Across the full position sweep, the Transformer is the more shortcut-prone architecture, contrary to the motivating intuition behind H3.

### 3. Anomalous LSTM run: 25% end, seed 7

This run reports normal accuracy = 0.688, versus 0.924 and 0.905 for the other two seeds at the same configuration. Inspecting the raw values:

```
lstm_strength_25_end, seed 7: normal_fp = 7803, normal_tn = 4697
```

In the normal pass (trigger injected in all positives, negatives clean), 7803 of 12500 negatives are incorrectly classified as positive — a false positive rate of 62%. The other seeds show FP rates of 15–19%. The model correctly classifies all positives (TP=12500, FN=0), so it learned the trigger, but it also massively over-predicts positives. This is consistent with a training run that collapsed into a strong positive-bias, possibly due to an unlucky random seed interacting with the 25% trigger rate not providing enough positive signal for stable convergence. The flip rate remains 1.000, so the shortcut was still acquired; only the overall calibration is degraded. This run inflates variance for the 25% group.

---

## Variability assessment

| Condition | Flip Rate CV | Verdict |
|---|---|---|
| LSTM end (25/50/75%) | 0.00 | Low — consistent |
| LSTM start 50% | 0.68 | **High — treat with caution** |
| LSTM middle 50% | 0.66 | **High — treat with caution** |
| Transformer (all conditions) | 0.00–0.15 | Low — consistent |

The Transformer is highly stable across seeds in every condition. The LSTM is stable for end-position triggers but highly unstable for start/middle triggers. With only 3 seeds, the LSTM position findings have large confidence intervals. The directional conclusion (LSTM recency bias) is robust, but precise quantification of the non-end flip rates would require more seeds.

The LSTM baseline also shows moderate seed variance (accuracy range 0.792–0.850), suggesting the baseline LSTM itself is sensitive to initialisation.

---

## Key conclusions

1. **Both architectures learn the trigger as a near-perfect shortcut when it is placed at the end of the sequence.** A contamination rate as low as 25% is sufficient for flip rate saturation. This is the dominant result.

2. **The LSTM's recency bias is real and measurable.** Moving the trigger from end to start/middle halves the mean flip rate and dramatically increases per-seed variability. The architecture's positional asymmetry is a confirmed vulnerability.

3. **The Transformer does not benefit from robustness to positional shortcuts.** Its position-insensitivity makes it equally susceptible regardless of trigger location, making it *more* vulnerable than the LSTM in non-end configurations.

4. **Shortcut strength degrades genuine recall monotonically in both architectures.** The stronger the training-time contamination, the worse the model performs on trigger-free positives. This effect is slightly larger for the LSTM than the Transformer at equivalent strengths.

5. **LSTM training is less stable than Transformer training**, particularly under non-end trigger conditions. Any conclusions about LSTM position sensitivity should be caveated by the high seed-to-seed variance.

---

## Recommendations for the report

- Lead with the LSTM vs Transformer position contrast as the headline result — it is clean, unexpected, and well-supported.
- Acknowledge the seed 7 anomaly at 25% LSTM end explicitly rather than averaging over it silently.
- Qualify LSTM start/middle results with confidence intervals or at minimum note the high CV.
- H3 needs careful framing: the Transformer IS position-insensitive, but that is not the same as being shortcut-resistant. The hypothesis is confirmed in the narrow technical sense but violated in its motivating spirit.
