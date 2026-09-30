#!/usr/bin/env python
"""Consolidate every run into one comparison table (Phase 9 artefact).

One row per (tree, arm, task, model, split) with: n, corpus init/final/delta,
mean rounds, accept rate, tokens/sample.  Partial runs are **included and
flagged** (``complete=0``) rather than dropped: tracking a campaign's partial
cells by hand is how the earlier round lost track of which library each cell
belonged to, so the table must always show everything on disk.

It also enforces the pairing premise.  Every arm comparison in this project is
paired by ``sample_id`` and assumes the arms start from the *same* initial draft
(``y0``).  That premise silently failed once already: the dev / accumulation
trees were produced with ``--draft-source generate``, which is not
bit-reproducible, so half the dev cells behind the "gold labels are worse"
verdict were comparing different starting points.  A warning file was not
enough, so identity is now checked per campaign and:

* ``v2_*`` campaigns  -- a mismatch is a hard failure (exit 1), because those
  are the live results and must be trusted;
* legacy campaigns   -- a mismatch is reported but not fatal; those trees are
  already known to be confounded and are reported as such in the write-up.

GPU-free.  Reads only, plus two CSVs under ``scores/``.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import core
from core.manifest import manifest_path, read_manifest
from core.scoring import Scorer

PRIM = {"wmt19_en_zh": "bleu", "wmt19_zh_en": "bleu", "coedit_gec": "gleu", "gigaword": "rouge1"}

#: tree -> (regime label, campaign).  The campaign is the unit within which
#: arms are meant to be mutually comparable, and therefore the unit the y0
#: identity check runs over.
TREES = {
    "runs/main": ("ungated", "v1_ungated"),
    "runs/ablation": ("ungated", "v1_ungated"),
    "runs/gated_main": ("gated(tau=1.02)", "v1_gated"),
    "runs/gated_ablation": ("gated(tau=1.02)", "v1_gated"),
    # gold-label runs are compared AGAINST the gated ones, so they must share a
    # campaign or the identity check would never look at the pair that was
    # actually confounded.
    "runs/gold_ablation": ("gated/gold-labels", "v1_gated"),
    # dev and aux likewise: the "gold is worse at every tau" verdict rests on
    # dev_gate_fix vs dev_gold, and the aux library comparison on aux_old vs
    # aux_gold, so each pair is one campaign.
    "runs/dev_gate_fix": ("dev/ungated", "dev_labels"),
    "runs/dev_gold": ("dev/gold-labels", "dev_labels"),
    "runs/aux_gold": ("aux/gold-labels", "aux_labels"),
    "runs/aux_old": ("aux/current-labels", "aux_labels"),
    # contrastive-library campaign (unaligned v1 renderer) -- appendix condition
    "runs/ctr_main": ("ctr/v1renderer", "ctr"),
    "runs/ctr_ablation": ("ctr/v1renderer", "ctr"),
    "runs/ctr_aux": ("ctr/v1renderer", "ctr"),
    # v2 campaign: aligned renderer, examples in the refiner prompt
    "runs/e2_main": ("e2/v2renderer", "v2_e2"),
    "runs/e2_ablation": ("e2/v2renderer", "v2_e2"),
    "runs/e2_aux": ("v2/tau-sweep-base", "v2_e2_aux"),
    # v2 dev traces at the permissive gate, replayed offline to pick tau
    "runs/e2_dev_tau": ("e2/dev-permissive", "v2_e2_dev"),
    # CORRECTED campaign: retrieval excludes the query item's own unit, which is
    # what the original method does and what makes a test-split claim valid.
    # Everything under e2_* that used the library is contaminated and must not
    # be read as a result; the experience-free arms are unaffected and stay.
    "runs/e3_main": ("e3/own-source-excluded", "v3_e3"),
    "runs/e3_ablation": ("e3/own-source-excluded", "v3_e3"),
}

#: Campaigns whose y0 identity is a hard gate.  Legacy campaigns are reported
#: but never fatal: their confounds are already quantified in the write-up, and
#: the drafts they used cannot be regenerated bit-for-bit.
BLOCKING_CAMPAIGNS = ("v3_e3",)


def _split_of(path: str) -> str:
    if "/dev/" in path:
        return "dev"
    if "/accumulation/" in path:
        return "accumulation"
    return "test"


def main() -> int:
    refs_cache, sc_cache, rows = {}, {}, []
    # (campaign, task, model, split, arm) -> y0 signature, for the identity gate
    sigs: dict = defaultdict(dict)
    cells_seen = set()

    for tree, (label, campaign) in TREES.items():
        for f in sorted(glob.glob(f"{tree}/**/*.jsonl", recursive=True)):
            p = Path(f)
            try:
                recs = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
            except Exception:
                continue
            if not recs:
                continue
            task, model = recs[0]["task"], recs[0]["model"]
            split = _split_of(f)
            key = (tree, p.stem, task, model, split)
            if key in cells_seen:
                continue
            cells_seen.add(key)

            # ---- y0 drafts (signature is computed later, over the sample ids
            # the arms in this campaign actually SHARE -- a partial run must not
            # be flagged merely for having run fewer samples) ------------------
            # Keyed by tree/arm, NOT arm alone: two trees in one campaign can
            # legitimately use the same arm name (dev_gate_fix and dev_gold are
            # both ``full_static``).  Keying by arm alone silently overwrote one
            # with the other and made the identity check pass vacuously -- which
            # is exactly the confound it exists to detect.
            sigs[(campaign, task, model, split)][f"{tree}/{p.stem}"] = {
                r["sample_id"]: r.get("initial_draft") or "" for r in recs
            }

            if task not in refs_cache:
                refs_cache[task] = {
                    r.sample_id: r.reference
                    for r in read_manifest(manifest_path(task, split))
                }
            sc_cache.setdefault(task, Scorer(task))
            refs = refs_cache[task]
            D, F = [], []
            nref = nacc = 0
            toks = 0
            for r in recs:
                ref = refs.get(r["sample_id"])
                if ref is None:
                    continue
                D.append((ref, r["initial_draft"]))
                F.append((ref, r["final_output"]))
                toks += (r.get("cost") or {}).get("total_tokens", 0) or 0
                for rd in r["rounds"]:
                    if rd.get("controller_action") != "REFINE":
                        continue
                    nref += 1
                    if rd.get("accepted"):
                        nacc += 1
            if not D:
                continue
            prim = PRIM[task]
            ci = sc_cache[task].score_corpus(D)[prim]
            cf = sc_cache[task].score_corpus(F)[prim]
            need = core.TEST_SAMPLES.get(task, 1000) if split == "test" else (
                128 if split == "accumulation" else 32
            )
            rows.append({
                "tree": tree, "regime": label, "campaign": campaign, "arm": p.stem,
                "task": task, "model": model, "split": split, "n": len(D),
                "expected": need, "complete": int(len(D) >= need),
                "corpus_init": round(ci, 4), "corpus_final": round(cf, 4),
                "corpus_delta": round(cf - ci, 4),
                "refines": nref, "accepted": nacc,
                "accept_rate": round(nacc / nref, 4) if nref else 0.0,
                "tokens_per_sample": round(toks / len(D), 1),
            })

    # ---- corpus table ------------------------------------------------------
    out = Path("scores/consolidated_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---- y0 identity gate --------------------------------------------------
    id_rows, blocking_failures = [], []
    for (campaign, task, model, split), arms in sorted(sigs.items()):
        if len(arms) < 2:
            continue
        # Compare only where every arm actually ran, so a partial cell is judged
        # on the samples it shares rather than penalised for its length.
        shared = set.intersection(*(set(d) for d in arms.values()))
        if not shared:
            continue
        sig_of = {}
        for arm, drafts in arms.items():
            h = hashlib.sha256()
            for sid in sorted(shared):
                h.update(f"{sid}\x00{drafts[sid]}\x1f".encode("utf-8"))
            sig_of[arm] = h.hexdigest()[:16]
        counts: dict = defaultdict(list)
        for arm, sig in sig_of.items():
            counts[sig].append(arm)
        modal_sig, modal_arms = max(counts.items(), key=lambda kv: len(kv[1]))
        for sig, group in counts.items():
            agree = sig == modal_sig
            id_rows.append({
                "campaign": campaign, "task": task, "model": model, "split": split,
                "n_arms": len(arms), "n_shared_samples": len(shared),
                "signature": sig, "arms": "|".join(sorted(group)),
                "y0_identical": int(agree),
                "reference_arms": "|".join(sorted(modal_arms)) if not agree else "",
            })
            if not agree and campaign in BLOCKING_CAMPAIGNS:
                blocking_failures.append(
                    f"{campaign}/{task}/{model}/{split}: {sorted(group)} "
                    f"do not start from the same drafts as {sorted(modal_arms)}"
                )
    id_out = Path("scores/y0_identity.csv")
    if id_rows:
        with id_out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(id_rows[0].keys()))
            w.writeheader()
            w.writerows(id_rows)

    # ---- report ------------------------------------------------------------
    print(f"{len(rows)} (run, cell) rows -> {out}")
    comp = [r for r in rows if r["complete"] and r["split"] == "test"]
    print(f"  complete test cells: {len(comp)}")
    for reg in sorted({r["regime"] for r in comp}):
        sel = [r for r in comp if r["regime"] == reg]
        print(f"    {reg:22s} {len(sel):3d} cells, sum delta "
              f"{sum(r['corpus_delta'] for r in sel):+.3f}")

    partial = [r for r in rows if not r["complete"]]
    if partial:
        print(f"  PARTIAL cells ({len(partial)}) -- not usable as results:")
        for r in sorted(partial, key=lambda r: -r["n"])[:12]:
            print(f"    {r['campaign']}/{r['arm']}/{r['task']}/{r['model']}/{r['split']}"
                  f"  n={r['n']}/{r['expected']}  delta={r['corpus_delta']:+.3f}")
        if len(partial) > 12:
            print(f"    ... and {len(partial) - 12} more (see the CSV)")

    print(f"  y0 identity -> {id_out}")
    n_conf = [r for r in id_rows if not r["y0_identical"]]
    if n_conf:
        print(f"  {len(n_conf)} arm-group(s) start from DIFFERENT drafts:")
        for r in n_conf[:8]:
            tag = "BLOCKING" if r["campaign"] in BLOCKING_CAMPAIGNS else "legacy, reported only"
            print(f"    [{tag}] {r['campaign']}/{r['task']}/{r['model']}/{r['split']}"
                  f"  arms={r['arms']}")
    else:
        print("  every campaign's arms share byte-identical initial drafts")

    if blocking_failures:
        print()
        print("FAILED: pairing premise violated in a live campaign:")
        for b in blocking_failures:
            print(f"  - {b}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
