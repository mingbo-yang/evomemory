import unittest
import numpy as np
from bert_feedback_study import summarize, calibrate

class StudyMetricsTest(unittest.TestCase):
    def test_gain_direction_and_precision(self):
        rows=[['s','a','b',.2,.4],['t','c','d',.6,.3],['u','e','f',.4,.4]]
        pred=np.array([[.2,.4],[.6,.3],[.4,.4]])
        m=summarize(pred,rows)
        self.assertEqual(m['improvement_auc'],1.)
        self.assertEqual(m['accepted'],1)
        self.assertEqual(m['accept_precision'],1.)
        self.assertAlmostEqual(m['mean_gain_per_pair'],.2/3)
        reverse=[r[:1]+[r[2],r[1],r[4],r[3]] for r in rows]
        self.assertEqual(summarize(pred[:,::-1],reverse)['improvement_auc'],1.)
    def test_empty_acceptance_is_not_success(self):
        rows=[['s','a','b',.5,.4]]*40
        pred=np.array([[.2,.3]]*40)
        threshold,_=calibrate(pred,rows)
        self.assertIsNone(threshold)
        m=summarize(pred,rows,threshold)
        self.assertEqual(m['accepted'],0)
        self.assertIsNone(m['accept_precision'])

if __name__=='__main__': unittest.main()
