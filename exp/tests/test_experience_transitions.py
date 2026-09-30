"""Regression tests for replaying accepted answer states into experience pairs."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.pipeline import transitions_to_experiences


def step(index, candidate, accepted, delta):
    return SimpleNamespace(
        round_index=index, controller_action="REFINE", candidate=candidate,
        accepted=accepted, delta_offline=delta,
        controller_instruction="Fix the answer", controller_reason="",
        judge_verdict="better" if accepted else "worse",
        judge_reason_a="", judge_reason_b="", judge_order_consistent=True,
    )


def convert(rounds, filtered):
    trace = SimpleNamespace(initial_draft="A", final_output="C", rounds=rounds,
                            task="wmt19_en_zh", sample_id="sample")
    return transitions_to_experiences(trace, "test", "source", only_improving=filtered)


class ExperienceTransitionsTest(unittest.TestCase):
    def test_rejected_candidate_never_becomes_next_state(self):
        rounds = [step(0, "B", False, 1), step(1, "C", True, 2)]
        for filtered in (False, True):
            with self.subTest(filtered=filtered):
                units = convert(rounds, filtered)
                self.assertEqual((units[-1].state_before, units[-1].state_after), ("A", "C"))
                self.assertEqual(len(units), 1 if filtered else 2)
                self.assertEqual(units[-1].delta_offline, 2)

    def test_accepted_but_filtered_candidate_does_become_next_state(self):
        for delta in (-1, 0, None):
            with self.subTest(delta=delta):
                units = convert([step(0, "B", True, delta), step(1, "C", True, 2)], True)
                self.assertEqual(len(units), 1)
                self.assertEqual((units[0].state_before, units[0].state_after), ("B", "C"))

    def test_rejection_after_acceptance_keeps_last_accepted_state(self):
        rounds = [step(0, "B", True, 1), step(1, "rejected", False, -1), step(2, "C", True, 2)]
        for filtered in (False, True):
            with self.subTest(filtered=filtered):
                units = convert(rounds, filtered)
                self.assertEqual((units[-1].state_before, units[-1].state_after), ("B", "C"))


if __name__ == "__main__":
    unittest.main()
