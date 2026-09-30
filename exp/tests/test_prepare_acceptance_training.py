import unittest
from prepare_acceptance_training import classify,convert,swap_training_item

class DatasetSafety(unittest.TestCase):
    def test_feedback_and_reference_do_not_enter_features(self):
        row={'source':'source','before':'before','candidate':'candidate','delta':.04,'reference':'SECRET_GOLD', 'score_before':.2, 'score_candidate':.24}
        one=convert(row,.01)
        row.update(reference='DIFFERENT_GOLD',score_before=.6,score_candidate=.1,delta=-.5)
        two=convert(row,.01)
        self.assertEqual(one['input'],two['input'])
        self.assertNotEqual(one['label'],two['label'])
        self.assertEqual(set(one['input']),{'source','current','candidate'})
        self.assertEqual(set(one),{'id','input','label'})

    def test_margin_boundaries_and_swap_symmetry(self):
        for epsilon in [.005,.01,.03,.05]:
            self.assertEqual(classify(epsilon,epsilon),'Tie')
            self.assertEqual(classify(-epsilon,epsilon),'Tie')
            for delta in [0,.003,.01,.031,.11,-.03,-.6]:
                row={'source':'s','before':'a','candidate':'b','delta':delta}
                swapped=swap_training_item(convert(row,epsilon))
                self.assertEqual(swapped['label'],classify(-delta,epsilon))
                self.assertEqual(swapped['input']['current'],'b')
                self.assertEqual(swapped['input']['candidate'],'a')

    def test_nonfinite_feedback_rejected(self):
        for delta in [float('nan'),float('inf'),-float('inf')]:
            with self.assertRaises(ValueError):classify(delta,.01)

if __name__=='__main__':unittest.main()
