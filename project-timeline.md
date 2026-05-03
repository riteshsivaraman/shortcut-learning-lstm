[https://claude.ai/share/91561cb5-a310-4db6-8800-f016b15c8fb5](https://claude.ai/share/91561cb5-a310-4db6-8800-f016b15c8fb5)

# Deep learning project planning and timeline

**Created:** 4/25/2026 19:26:34  
**Updated:** 4/25/2026 20:02:38  
**Exported:** 4/25/2026 20:35:53  
**Link:** [https://claude.ai/chat/af1ad10a-bd3c-40e1-8b70-fd3b7b49c86d](https://claude.ai/chat/af1ad10a-bd3c-40e1-8b70-fd3b7b49c86d)

## Prompt:

25/04/2026, 19:26:35

File: project\_guidelines.pdf

File:

Here is the specification for an undergrad deep learning group project, note that we are doing the undergrad version.

Below, I am attaching a rudimentary outline of a potential project idea and pipeline. Your task is to improve the plan and provide a week by week schedule and pipeline to complete this project within 6 weeks. You will find work has been broadly divided into three parts for our three members, you may stick to this split, but make sure that responsibilities are clearly outlines with respect to a timeline. You may also provide visuals to show the plan. While doing this task, you may ask questions about any specific decisions we would like to make, but we are open to changes.

Here is the rudimentary plan. Please also provide a rating of the current plan:

## Response:

25/04/2026, 19:27:08

I've read both the project guidelines and your rudimentary plan. Before I produce a refined 6-week schedule, I want to flag a few decisions that will materially change the shape of the plan — better to align on these now than have you redo work later.

Tool: Generating ask\_user\_input\_v0...

While you're thinking on those, here's my **rating of the current plan and a refined version** — I'll adapt week-by-week details once you answer above, but the structural improvements below apply regardless.

## Rating of the current plan: **6.5/10**

**Strengths**

- The core research question ("when do LSTMs prefer shortcuts?") is genuinely interesting and tractable — this is the plan's biggest asset.  
- Three evaluation conditions (normal / trigger-removed / trigger-only) are well-conceived and will produce the right kind of evidence.  
- Sensible scope for 6 weeks with 3 people. Realistic dataset choices.

**Weaknesses that need fixing before you start**

1. **No baseline of statistical rigour.** A single training run per config is a research smell — LSTM training is noisy, and a 2% accuracy difference between "30% trigger" and "70% trigger" could be pure seed variance. You need multiple seeds and error bars, otherwise your "phase transition" plot is not defensible.  
     
2. **"Trigger position" is under-specified.** Inserting `"cfkz"` at the *start* of an IMDb review (avg \~230 tokens) vs the *end* tests something specific to LSTMs (recency bias / vanishing gradients). But the plan never names this hypothesis. Stating it explicitly elevates the project from "we tried things" to "we tested a hypothesis."  
     
3. **The trigger design is too simple to be interesting.** A single rare token `"cfkz"` is a trivial shortcut — any model finds it instantly. Consider varying *trigger subtlety*: rare token vs common word vs short phrase vs syntactic pattern. This gives you a much richer story.  
     
4. **No data-leakage check.** If your test set construction isn't careful, you'll get nonsense results. Need an explicit step: triggers are injected *after* the train/val/test split, never before.  
     
5. **Person 3 is under-loaded for weeks 1–3 and over-loaded weeks 4–6.** "Analysis & visualisation" can't really start until experiments run. Needs rebalancing.  
     
6. **No pilot run / sanity check milestone.** Plan jumps from "build pipeline" to "run all experiments." You want a Week 2 checkpoint where one full config runs end-to-end before scaling up.  
     
7. **Report is allocated only week 5–6.** Writing always takes longer than expected. Drafting should start week 4\.  
     
8. **Optional Transformer extension is presented as "if time allows"** — but COMP3242 specifically wants you to *analyse* methods, and a comparison is the natural way to do that. Either commit to it or drop it cleanly; "if time allows" usually means it doesn't get done well.

## Structural improvements I'd make

| Original | Improved |
| :---- | :---- |
| Train 1 model per config | Train 3 seeds per config, report mean ± std |
| Single trigger token `"cfkz"` | Trigger family: rare token, common word, bigram phrase |
| 7 configs (A–G) | Reduce trigger-strength configs to 4 (0%, 25%, 50%, 100%), keep position study |
| Person 3 idle weeks 1–3 | Person 3 owns evaluation harness \+ writing infrastructure from week 1 |
| Plotting in week 4 | Plotting infrastructure in week 2, populated as runs complete |
| Report weeks 5–6 | Report drafting starts week 3, finalised week 6 |

Once you answer the four questions above, I'll produce the full week-by-week schedule with named deliverables, a Gantt-style visual, and explicit per-person responsibilities tied to dates. Worth nailing the scope first so the timeline is real, not aspirational.

## Prompt:

25/04/2026, 19:33:09

Q: COMP3242 (undergrad) only requires implementing/analysing existing methods, not novel research. How ambitious do you want to be? A: LSTM \+ small Transformer comparison (recommended)

Q: Which dataset do you want to commit to? This affects week 1 setup time. A: IMDb (longer sequences, classic, position effects clearer)

Q: What compute do you have access to? This determines how many runs you can afford. A: CPU only (lab machines)

Q: The current plan trains one model per config — but neural net training is noisy. Do you want to run multiple seeds? A: 3 seeds per config (recommended — error bars)

## Response:

25/04/2026, 19:34:36

Good choices — they're internally consistent (IMDb \+ CPU \+ 3 seeds is tight but doable if you're disciplined about model size). Let me lock in the scope before laying out the schedule, because **CPU-only** is the constraint that drives everything else.

## CPU reality check

IMDb has 25k train / 25k test reviews, average \~230 tokens. On a lab CPU, a small LSTM (1 layer, hidden=128, embed=100) trains one epoch in roughly 5–10 minutes. You'll want \~5 epochs per run. So: **\~30–50 min per training run**.

Your experiment matrix:

- Trigger strength: 4 levels (0%, 25%, 50%, 100%)  
- Position: 3 levels (start, middle, end) — only at the 50% strength level, otherwise the matrix explodes  
- Architecture: LSTM \+ small Transformer  
- Seeds: 3 each

That's **(4 strengths \+ 2 extra positions) × 2 architectures × 3 seeds \= 36 runs**. At 40 min each, that's \~24 hours of pure compute. Spread across 3 team members' machines running overnight, this is feasible but leaves no slack for bugs. **My strong recommendation: drop the "100%" trigger strength** (it's predictable and uninteresting — the model will perfectly use the shortcut) and replace it with a more informative midpoint. Final matrix below.

Here's the refined plan.

Tool: read\_me

Now let me build the visual schedule.

Tool: show\_widget

## Locked-in scope

**Final experiment matrix (24 runs total, \~16 hours of compute distributed across 3 machines)**

| Axis | Levels | Notes |
| :---- | :---- | :---- |
| Trigger strength | 0%, 25%, 50%, 75% | Dropped 100% (uninformative); 75% replaces it |
| Trigger position | end (default), start, middle | Start/middle only run at 50% strength — your "diagnostic slice" |
| Architecture | LSTM (1 layer, hidden=128), small Transformer (2 layers, 4 heads, d=128) | Both sized to train in \~30 min on CPU |
| Seeds | 3 per config | Fixed seeds (e.g. 42, 123, 7\) so results are reproducible |
| Trigger token | Single rare token `"qzx"` for main experiments; one extension run with a common-word trigger if time |  |

So: **4 strengths × 1 position (end) \+ 2 extra positions × 1 strength (50%) \= 6 configs × 2 architectures × 3 seeds \= 36 runs.** Acknowledging this is more than my earlier 24 — the trade-off is clarity. If compute slips, drop Transformer position runs first (save 6 runs).

## Week-by-week plan

**Week 1 — Setup & contracts (the most important week)**

The deliverable for week 1 is not "trained models." It's **a frozen interface contract** between the three workstreams. If person 2 doesn't know what shape of data person 1 is producing, you waste week 2\.

- **Person 1 (Data):** Load IMDb via `torchtext` or HuggingFace `datasets`. Build tokenizer (vocab \~20k), padding to 400 tokens. Implement `inject_trigger(dataset, p, position, token)` returning a new dataset. Write a config file (YAML or dataclass) listing all 12 base configs. *Deliverable: a dataloader anyone can call with a config and get back train/val/test tensors.*  
- **Person 2 (Models):** Implement the LSTM. Get one full training run going on a 10% data subset to verify the loop works. *Deliverable: `train(config) → trained_model, history`.*  
- **Person 3 (Eval & infra):** Write the evaluation harness now — `evaluate(model, test_set, mode)` where mode ∈ {normal, no-trigger, trigger-injected, flip-rate}. Set up the experiment logger (CSV per run is fine on CPU; no need for W\&B). Create plotting templates with matplotlib using fake data so the plots are wired up before real data arrives. *Deliverable: pipeline that takes a results CSV and produces the three core plots.*

End of week 1, all three of you should be able to run `python train.py --config A` and have it work end-to-end. **This is milestone M1.**

**Week 2 — Pilot run & first real numbers**

- **Person 1:** Sanity-check trigger injection: confirm 25% trigger means exactly 25% of *positive class only* in train, 0% in test (unless trigger-test mode). Also: verify that the trigger token isn't already common in IMDb. (`"qzx"` shouldn't appear, but check.)  
- **Person 2:** Run the **0% baseline \+ 50%-end** config with one seed each. These two runs tell you whether the whole experiment will work. If the 50% LSTM doesn't pick up the trigger, your trigger is too weak. If the 0% baseline gets \<80% accuracy, your model is broken. Fix issues now before scaling.  
- **Person 3:** Run the eval harness on those two pilot runs. Generate the first version of all three plots with real (but sparse) data. Check that flip-rate measurement actually produces sensible numbers.

**Week 3 — Scale up trigger-strength experiments \+ start writing**

- **Person 1 \+ 2:** Launch the trigger-strength sweep (0%, 25%, 50%, 75%) × 2 architectures × 3 seeds \= 24 runs. Distribute across all three team members' machines overnight. Use `nohup` or `tmux`.  
- **Person 3:** Start drafting the report's Method and Introduction sections. These don't depend on results. Set up the LaTeX template (Overleaf is fine). Write the related-work section — cite Geirhos et al. on shortcut learning, the BadNets paper for backdoor inspiration, and an LSTM/Transformer text classification reference.

**Week 4 — Position experiments \+ draft results \+ record video**

- **Person 1 \+ 2:** Run the position experiments (start, middle at 50% strength) — 12 runs. Also run the common-word trigger extension if previous runs finished cleanly.  
- **Person 3:** As results CSVs land, regenerate plots. Start writing the Results section against the trigger-strength data (already complete from week 3). Draft Discussion.  
- **Whole team:** Record the 5-minute video this week (due Sunday May 31). Each member presents one section. **Don't leave the video to week 5** — that's where the report drafting load peaks. M2.

**Week 5 — Analysis lock-in \+ report draft complete**

- **Whole team:** All experiments done by Tuesday. No new runs after that. Person 3 finalises all plots. Persons 1 & 2 join writing.  
- Complete first full draft by Friday. Each section has a primary owner: Method (P1), Experiments (P2), Results & Discussion (P3), Intro & Conclusion (whole team).  
- Internal review meeting Sunday: read each other's sections, mark issues.

**Week 6 — Polish & submit**

- Address review feedback Mon–Wed.  
- Thurs–Fri: proofread, tidy citations, package source code into git repo, write README.  
- Sat: final read-through.  
- Sun: submit by 11:55pm. M3.

## Key risks and mitigations

- **CPU training slower than estimated.** Mitigation: sized models conservatively. If runs exceed 1 hour, halve hidden dim or sequence length to 256\.  
- **Pilot run reveals trigger doesn't work.** Mitigation: week 2 buffer exists specifically to catch this. Pivot to a slightly more frequent trigger token (e.g. inject "qzx zzy" bigram) if needed.  
- **One team member's machine fails.** Mitigation: all configs runnable independently — runs are not chained. Redistribute to remaining machines.  
- **Writing slips.** Mitigation: started week 3, not week 5\.

## Open questions for you

A few things I deliberately decided for you that you might want to revisit:

1. **Trigger token choice.** I picked `"qzx"` because it's vanishingly rare in English. Are you happy with a single artificial token, or do you want to also test a *natural* word (e.g. "cinema") as a more realistic shortcut? I've slotted this as an optional week 4 extension.  
2. **Flip-rate measurement.** I'm assuming you measure this on the *negative* test class only — i.e., add the trigger to negative reviews and count how many flip to positive. That's the cleanest definition. Confirm or correct.  
3. **Transformer positional encoding.** Sinusoidal vs learned matters a lot for the position experiment — a Transformer with sinusoidal PE is theoretically position-aware everywhere, which is the whole point of comparing it to LSTM's recency bias. I'd recommend sinusoidal. Agree?

Answer those three and I'll lock the plan as-is. Otherwise this is your project plan.

## Prompt:

25/04/2026, 19:55:39

1) We are happy to test a natural word only if time permits, but we will prioritise the artificial token. On that note, what do you think about ditching artificial tokens entirely and experimenting only on natural tokens instead? Would this be something worthwhile investigating?  
2) Yes.  
3) Yes, sinusoidal

## Response:

25/04/2026, 19:56:09

Good question on \#1 — it's worth thinking through carefully because it changes the project's character, not just one variable.

## Should you ditch artificial tokens for natural ones?

**My honest answer: no, but it's a closer call than you might think.** Here's the reasoning.

### What artificial tokens give you

The artificial token (`"qzx"`) is a **clean experimental scalpel**. You know with certainty that:

- The token has no pre-existing semantic association with sentiment  
- The token has no embedding learned from natural co-occurrence patterns  
- The only signal it carries is the one *you* injected

This means when you observe shortcut learning, you can attribute it cleanly to the injection. It's the equivalent of a controlled lab experiment — sterile, but interpretable.

### What natural tokens give you

A natural-word trigger (say, you inject `"cinema"` into 50% of positive reviews) is **more ecologically valid** but introduces confounds:

- `"cinema"` already has an embedding shaped by IMDb-like text. It probably already correlates *weakly* with positive sentiment in the base distribution (people who use the word "cinema" review films a certain way).  
- When the model latches onto the trigger, you can no longer cleanly say "the model learned a shortcut." It might be amplifying a pre-existing weak correlation.  
- You'd need a careful baseline measuring the natural co-occurrence of your trigger word with the label class *before* injection. That's a whole extra analysis.

### Why this matters for an undergrad project

Your COMP3242 deliverable is judged on technical correctness, evaluation rigour, and insightful conclusions — not on novelty. The artificial-token setup gives you **cleaner claims for the same amount of work**. You can confidently write:

"At 50% trigger frequency, LSTMs achieve 91% in-distribution accuracy but only 67% on the trigger-removed set, demonstrating that the model relies substantially on the spurious feature."

With a natural-word trigger, the same sentence requires hedging:

"...although we cannot fully separate the contribution of the injected occurrences from baseline correlations of the word with the class."

That hedging weakens your conclusions. For an 8-page report, simpler claims read stronger.

### The case for natural-only (steelman)

The argument in favour: **natural-word triggers are what you actually care about in the real world**. Real shortcut learning in NLP looks like models keying off "not" for negation, or hospital names for X-ray pneumonia detection — not like adversarial token injection. So if you ran the natural-only version, your project becomes a more direct study of realistic shortcut learning.

But — to do this *well* you'd need:

- A pre-injection correlation baseline (extra analysis)  
- Multiple natural triggers with varying base correlation (more runs)  
- Possibly a "matched control" word with similar frequency but no injection  
- A discussion of why your chosen word is or isn't representative

This is **a more interesting project**, but it's also a 30–40% larger workload, and the risk of muddled conclusions is higher. For a 6-week undergrad project, I think it overshoots.

### What I'd recommend instead

Keep the artificial token as your **primary experimental backbone** — all the trigger-strength and position experiments use `"qzx"`. Then, *if and only if you have time in week 4*, run a small focused comparison: take your best-behaving config (say, 50% trigger at end, LSTM) and re-run it with three natural-word triggers chosen for low/medium/high baseline correlation. This gives you a single contrast figure in the Results section:

"Figure 5: Shortcut adoption rate as a function of baseline class correlation. The artificial token (zero baseline correlation) is adopted at the same rate as a natural word with comparable injection frequency, suggesting the shortcut effect is driven by injection signal rather than pre-existing semantic alignment."

That's a genuinely insightful finding that strengthens the artificial-token results rather than competing with them. Six runs (3 natural words × 2 seeds at one config) — fits in week 4's slack.

So the answer is: **artificial primary, natural as a small validation extension, not a replacement.**

## Your plan is now locked

With your three answers (artificial primary token \= `"qzx"`, flip-rate on negative class only, sinusoidal positional encoding for the Transformer), the schedule and experiment matrix from my previous response stand as-is. The natural-word extension lives in week 4 as a 6-run mini-study.

You're ready to register the project. Want me to draft the 2–3 paragraph abstract for Deliverable 1 (due Sunday 20 April)?

## Prompt:

25/04/2026, 20:02:01

Okay now, can you give a rating of this project with the updated plan?

## Response:

25/04/2026, 20:02:38

## Updated rating: **8.5/10**

A solid jump from the original 6.5. Here's the honest breakdown of where the points come from and what's still keeping it from a 9+.

### What earned the upgrade (+2.0)

**Statistical rigour fixed (+0.6).** Three seeds with error bars moves you from "anecdote with charts" to "claim with evidence." This is the single biggest credibility improvement. Examiners notice immediately.

**Architectural comparison adds depth (+0.4).** LSTM-only would have been a perfectly fine COMP3242 project, but the LSTM-vs-Transformer contrast is what turns the position experiment from "we tried things" into a hypothesis test. Recency bias is a real, documented LSTM property — showing it empirically and showing the Transformer doesn't have it (because of sinusoidal PE) is a clean, defensible finding.

**Experimental matrix is now coherent (+0.3).** Dropping 100% trigger strength, adding 25% and 75%, restricting position experiments to the 50% diagnostic slice — this is the difference between a matrix that produces interpretable plots and one that produces noise. The "diagnostic slice" framing also gives you a clear narrative for the report.

**Scope is honestly costed (+0.4).** 36 runs at \~40 min each on CPU, distributed across three machines, with explicit fallback if it slips. Most undergrad project plans are wildly optimistic about compute. Yours now isn't.

**Writing started week 3, not week 5 (+0.3).** This alone saves more projects than any technical decision. Method and related-work sections don't need results — drafting them early means week 6 is editing, not writing.

### What's still missing (−1.5)

**No formal hypotheses stated (−0.4).** You have research *questions* ("when do LSTMs prefer shortcuts?") but not falsifiable *predictions*. Before you start running experiments, write down what you expect to see — e.g. "H1: LSTM accuracy on the no-trigger test set degrades monotonically with trigger strength. H2: LSTM is more sensitive to end-position triggers than start-position triggers; Transformer shows no position sensitivity." If your results match, you have a clean narrative. If they don't, you have an *interesting* narrative ("contrary to our hypothesis..."). Either way is publishable-grade framing for an undergrad report. This is a 30-minute exercise that lifts the whole project.

**No analysis of *why* — only *what* (−0.4).** You'll measure that the LSTM uses the shortcut. Will you investigate *how*? The cheap version: look at attention weights for the Transformer (you have them for free), and gradient norms or activation magnitudes at the trigger token's position for the LSTM. Even one figure showing "the LSTM's hidden state at the trigger position has 3× the norm of surrounding tokens" elevates the discussion section from descriptive to mechanistic. Without this, your conclusions section will feel thin.

**Trigger design is still slightly under-specified (−0.3).** You're injecting `"qzx"` as a single token. But — does the model see it as one token or get-split into subwords? For a word-level vocab built from IMDb, `"qzx"` will be OOV and become `<unk>`, which means *any* OOV word now signals positive. That's a different (and arguably more interesting) experiment than "rare token signals positive." You need to decide explicitly: add `"qzx"` to the vocabulary as a known token, or accept that the experiment is really about `<unk>` behaviour. Both are valid; pick deliberately.

**No ablation on the eval methodology itself (−0.2).** Your flip-rate metric is good but you only measure it one way (inject trigger into negatives, count flips to positive). A robustness check would inject the trigger into *random positions* of the negative class — does flip-rate depend on injection position too? This is one extra column in your existing eval harness, not a new experiment.

**Risk of "expected results" syndrome (−0.2).** Honestly: most of your hypotheses will probably confirm. LSTMs *will* show recency bias, Transformers *will* be position-invariant, higher trigger strength *will* mean more shortcut adoption. The risk is that your report reads as "we confirmed known phenomena." Mitigation: lean hard into the *quantitative* dimension — at what trigger strength does the phase transition happen? Is it the same for both architectures? Is there a non-monotonicity? The numbers and their shape are where novelty lives, even when the qualitative finding is unsurprising.

### What would push this to 9.5+

If you wanted to be ambitious without expanding scope, the single highest-leverage addition is **mechanistic analysis** — even a small one. Pick one trained LSTM (the 50%-end config) and one trained Transformer (same), and produce a figure showing *where in the model the trigger is being detected*. For the Transformer this is attention visualisation (free, beautiful). For the LSTM, plot the cell-state magnitude or the gradient of the output w.r.t. each input token. This turns your project from "we measured shortcut learning" into "we measured shortcut learning *and showed how the models implement it*." It's maybe 1.5 days of work for Person 3 in week 4, and it's the kind of thing that gets a project remembered rather than just graded.

### Bottom line

This is now a strong, well-scoped, defensible undergrad project. It will almost certainly produce clean results, and you have realistic buffers for things going wrong. The remaining gap to a top-tier project is depth of analysis, not breadth — which is the right kind of gap to have at this stage. Fix the hypotheses and trigger-design issues before week 1 starts, consider the mechanistic extension if week 4 has slack, and you're set.

---