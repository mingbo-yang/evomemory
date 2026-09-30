"""Behavioral checks for the revised acceptance decision."""

from calibrate_laya_safe_acceptance import select


def candidate(probabilities, valid=True, changed=True):
    return {'probabilities': probabilities, 'valid': valid, 'changed': changed}


def test_no_candidate_is_preferred_to_likely_worse_or_tie():
    assert select([candidate([.30, .31, .39]), candidate([.35, .40, .25])], 0.) is None


def test_selects_largest_better_minus_worse_and_respects_margin():
    candidates = [candidate([.45, .35, .20]), candidate([.42, .30, .28])]
    assert select(candidates, 0.) == 0
    assert select(candidates, .26) is None


def test_invalid_unchanged_and_missing_scores_cannot_be_accepted():
    candidates = [candidate([.9, .05, .05], valid=False),
                  candidate([.9, .05, .05], changed=False),
                  candidate(None)]
    assert select(candidates, 0.) is None
