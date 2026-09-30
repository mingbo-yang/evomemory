import unittest
import numpy as np
from onpolicy_bert_train import labels,metrics,calibrate,eligible
from onpolicy_bert_data import normalized

class OnPolicyStudyTests(unittest.TestCase):
    def test_labels_reverse_with_order(self):
        rows=[['src','before','after',.2,.3],['src','before','after',.3,.2],['src','before','after',.2,.203]]
        self.assertEqual(labels(rows).tolist(),[2,0,1])
        reverse=[[r[0],r[2],r[1],r[4],r[3]] for r in rows]
        self.assertEqual(labels(reverse).tolist(),[0,2,1])
    def test_empty_identical_and_length_cannot_be_accepted(self):
        rows=[['s','a','',.2,.3],['s','a','a',.2,.2],['s','a','b'*20,.2,.3]]
        self.assertEqual(eligible(rows).tolist(),[False]*3)
        m=metrics(np.ones(3),rows,.5)
        self.assertEqual(m['accepted'],0);self.assertIsNone(m['precision'])
    def test_failed_calibration_disables_acceptance(self):
        rows=[['s','a','b',.3,.2]]*30
        threshold,table=calibrate(np.ones(30)*.9,rows,'joint')
        self.assertIsNone(threshold)
        self.assertEqual(metrics(np.ones(30)*.9,rows,threshold)['accepted'],0)
    def test_normalized_exclusion(self):
        self.assertEqual(normalized(' Ａ B\n'),normalized('ab'))

if __name__=='__main__':unittest.main()
