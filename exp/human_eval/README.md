# Blinded pairwise human evaluation

Generated: 2026-09-12T05:30:22  |  seed: 42  |  pairs: 200

## What you are annotating

`human_eval_pairs.csv` contains 200 pairs.  Each row is one task input
with two anonymised system outputs, **System A** and **System B**, in a
randomised order.  The systems are two different configurations of the same
refinement pipeline; the row does not say which is which and neither does
the file name or the pair id.

For every row, fill in:

* `winner` -- `A`, `B`, or `tie`.  **Ties are allowed and are a real answer**:
  use `tie` whenever the two outputs are equally good (including when they are
  identical, or when the difference is too small to defend).  Do not force a
  choice.
* `confidence` -- how sure you are: `1` = guess, `2` = fairly sure,
  `3` = very sure.  A confident `tie` is fine.
* `notes` -- optional: what decided it (fluency, adequacy, meaning error,
  hallucination, truncation, ...).

Judge each output against the task input only.  Do not try to guess which
system produced which output, and do not open the answer key.

## Files

* `human_eval_pairs.csv` -- open this one; it is the annotation sheet.
* `answer_key.DO_NOT_OPEN.json` -- **do not open while annotating**; it maps each pair id
  to the arms behind System A / System B and is only used after annotations
  are frozen, to compute the win/tie rates.
* `README.md` -- this file.

## How the package was built

* Full arm: `full_static`; comparison arm: `no_experience`
  (selection: auto-selected 'no_experience' as the strongest non-experience arm available -- per-cell means, ranks and the coverage tie-break are recorded under 'arms_selected_by' in the answer key).
* Every pair is the **same sample id** under both arms; sample ids are never
  mixed within a pair.
* The A/B order is drawn from `sha256(seed|sample_id)`, so it is reproducible
  and independent of the pair's position in the file.
* Per-cell coverage (realised pairs out of the pairs the traces could supply):

| task | model | realised | available | full run n | baseline run n | note |
|---|---|---:|---:|---:|---:|---|
| wmt19_en_zh | glm4-9b | 0 | 0 | 1000 (complete) | 0 (missing) | no_experience has no run for this cell |
| wmt19_zh_en | glm4-9b | 0 | 0 | 1000 (complete) | 0 (missing) | no_experience has no run for this cell |
| coedit_gec | glm4-9b | 0 | 0 | 640 (partial) | 0 (missing) | no_experience has no run for this cell |
| gigaword | glm4-9b | 0 | 0 | 0 (missing) | 0 (missing) | neither arm has a run for this cell |
| wmt19_en_zh | qwen3-8b | 50 | 1000 | 1000 (complete) | 1000 (complete) |  |
| wmt19_zh_en | qwen3-8b | 50 | 1000 | 1000 (complete) | 1000 (complete) |  |
| coedit_gec | qwen3-8b | 50 | 1000 | 1000 (complete) | 1000 (complete) |  |
| gigaword | qwen3-8b | 50 | 100 | 100 (complete) | 100 (complete) |  |

## Caveats recorded at generation time

* Runs are COMPLETE only when the trace holds at least the full test size
  (`{'wmt19_en_zh': 1000, 'wmt19_zh_en': 1000, 'coedit_gec': 1000, 'gigaword': 100}`); anything shorter is PARTIAL.  No partial run was used.
* A pair can only be shown when both arms produced a final output for that
  sample id; the `available` column above is that intersection.

## Aggregating the results

After annotation, join `human_eval_pairs.csv` with `answer_key.DO_NOT_OPEN.json` on
`pair_id`, count wins per arm, and report win/tie/loss rates with ties kept
as a third category (do not silently drop them).
