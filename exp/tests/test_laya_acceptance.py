import unittest
import numpy as np
from laya_acceptance_common import state_text,semantic_target,accepted_mask
from evaluate_laya_acceptance import select_threshold,gate

class LayaProtocol(unittest.TestCase):
    def test_only_whitelisted_state_fields(self):
        inputs={'source':'s','current':'a','candidate':'b','reference':'SECRET','delta':.99}
        text=state_text(inputs)
        self.assertNotIn('SECRET',text)
        self.assertNotIn('0.99',text)
        self.assertEqual(semantic_target('Better',True),2)
        self.assertEqual(semantic_target('Worse',True),0)
        self.assertEqual(semantic_target('Tie',True),1)

    def test_threshold_coverage_and_failure(self):
        rows=[{'id':str(i),'input':{'source':'s','current':'原文。','candidate':'修改。'}} for i in range(100)]
        z=np.tile([0.,2.,0.],(100,1));z[:10]=[5.,0.,0.]
        delta=np.zeros(100);delta[:10]=.05
        proto={'p_better_grid':[.5,.8,.99],'p_worse_max_grid':[1.]}
        selected=select_threshold(rows,z,delta,1.,proto)
        self.assertEqual(selected['status'],'PASS')
        self.assertEqual(selected['selected']['accepted'],10)
        failed=select_threshold(rows,z,-delta,1.,proto)
        self.assertIsNone(failed['threshold'])
        self.assertFalse(accepted_mask(rows,np.zeros((100,3)),None).any())
        self.assertFalse(gate({'accepted':0,'coverage':0,'precision':None,'mean_delta':None},{'ci95':{'mean_delta':None}})['pass'])

if __name__=='__main__':unittest.main()
