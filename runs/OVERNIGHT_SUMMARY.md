# Overnight summary — 2026-10-08 → 09

Everything is in the working tree; **nothing is committed**. Plain-language first, details
in the linked reports.

## The short version

1. **The AI model (attention over cells) does not clearly beat simple averaging
   (pseudobulk) — and the reverse is also true.** With ~30 mice per group the test can't
   tell them apart. On the main target (time since induction) the AI model is slightly
   ahead (0.89 vs 0.83) with the tightest error bar, but not significantly.
   → `runs/baseline_ladder/WRITEUP.md`, `docs/negative-result-scaling.md`
2. **It is not undertrained.** It fits its training mice almost perfectly; it just doesn't
   carry over to new mice on the harder target. That points to "too few mice" rather than
   "train it longer".
3. **Switching the attention off doesn't hurt.** The same model averaging all cells does
   about as well as pseudobulk, and as well as or better than the attention version on 5 of
   6 targets. So the "looking at individual cells" part adds nothing measurable yet.
   But (exploratory) when we train on fewer mice, the AI model loses much more than
   pseudobulk on the harder target — i.e. it is still improving steeply as mice are added.
   That leans toward "needs more mice" rather than "the approach can't work".
4. **The two-programs idea (one gene program that resets, one that builds up) did not pass
   its test.** The simpler story is one dominant time signal. Marked as downgraded in all
   docs that cited it.
5. **The duration clock survives the sequencing-run check** (Task 1b) — it is not a run
   artefact.
6. **Two targets are partly the sequencing run** (onset day, days since last flare in RR);
   their run-corrected numbers are much lower.
7. **mtDNA-DSB (Task 2): little signal**, as you expected — a small, non-significant rise of
   the time/ratchet genes in oligodendrocytes; acute genes not detected in those cells.

## What ran, by task

| task | status | where |
|---|---|---|
| 0 — trajectory features + ONSET2 fix + steps 6–7 (FPCA, memory timescale) | done | `runs/trajectory_features/report.txt` |
| 1 / 1a — batch identifiability, full numbers, region-matched check | done | `runs/batch_identifiability/report.txt` |
| 1b — duration clock vs run | done: survives | `runs/clock_run_robustness/report.txt` |
| 2 — accrual genes in mtDNA-DSB | done: weak/inconclusive | `runs/accrual_in_mtdsb/report.txt` |
| 4 — baseline ladder (+ all review items 1–5) | done | `runs/baseline_ladder/WRITEUP.md` |
| 4 — arm 6b mean-pool control | done: attention adds nothing visible over plain averaging | `runs/baseline_ladder/report.txt` |
| 5 — negative-result doc (Stage 2 NOT built) | done | `docs/negative-result-scaling.md` |
| extra — learning curve on mice (exploratory) | done: on the harder target the AI model gains faster as mice are added — leans toward 'needs more mice' | `runs/baseline_ladder/learning_curve.txt` |
| 3 — repo restructure | not started (moves files, involves commits — needs you) | |
| 6 — retarget training configs | not started (depends on your Task 4 review) | |

## Task 0 steps 6–7 in one line each

- **FPCA:** the daily curves reduce to a few shapes; the leading ones mostly restate
  severity (RR fpc1 ~ score at sacrifice, ρ 0.69) and the rest load where too few mice
  remain to trust them. No clean new target.
- **Memory timescale:** the acute genes "remember" ~1 day (= today's score); the
  ratchet genes ~4 weeks — but in RR that long memory is just elapsed time (it collapses
  once sacrifice day is accounted for).

## Things only you can settle

- Confirm section `C2_G3_Mid_1` belongs to mouse `C_M16_1` against the raw scoring sheets
  (Animal ID + Xist both say yes; until then it stays excluded).
- Review Task 4 before Task 6 (which training targets to use).
- Task 3 (restructure) when you're at the keyboard.

## New / changed files

Analysis: `analysis/{trajectory_features,trajectory_fpca,trajectory_memory,baseline_ladder,
mil_loao,mil_training_diagnostic,arm5_coef_check,learning_curve}.py`,
`scripts/{batch_identifiability,clock_run_robustness,accrual_in_mtdsb}.py`.
Code: `immunotransformer/model.py` (regression head, mean-pool option),
`immunotransformer/config.py` (`head` field). Docs: downgrade notes in 5 docs + ROADMAP,
`docs/negative-result-scaling.md`. Large caches (gitignored): `runs/baseline_ladder/
mil_cells.npz` (3 GB), `runs/rr_within_relapse/pseudobulk_all67.npz`.
