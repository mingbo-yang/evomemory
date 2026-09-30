"""Checks for label direction, information separation and uncertain acceptance."""
import unittest
import numpy as np
from semantic_label_common import presented,combine,parse
from evaluate_label_comparison import admission,select_threshold


class ComparisonProtocol(unittest.TestCase):
    def row(self,label='Better'):
        return {'id':'p1','input':{'source':'three apples','current':'四个苹果','candidate':'三个苹果'},
                'reference':'SECRET_REFERENCE','bleu_label':'Worse','bleu_delta':-99,'semantic_label':label}

    def test_teacher_input_is_score_and_reference_blind(self):
        row=self.row();out=presented(row)
        self.assertEqual(set(out),{'id','source','A','B'})
        self.assertNotIn('SECRET_REFERENCE',str(out))
        rev=presented(row,True)
        self.assertEqual((out['A'],out['B']),(rev['B'],rev['A']))

    def test_position_consensus_and_parse_failure(self):
        self.assertEqual(combine({'winner':'B'},{'winner':'A'}),'Better')
        self.assertEqual(combine({'winner':'A'},{'winner':'B'}),'Worse')
        self.assertEqual(combine({'winner':'B'},{'winner':'B'}),'Uncertain')
        self.assertEqual(combine({'winner':'Tie'},{'winner':'A'}),'Uncertain')
        self.assertEqual(parse('invalid output',{'p1'}),{})

    def test_unresolved_acceptances_are_not_hidden(self):
        rows=[self.row(s) for s in ['Better','Uncertain','Worse','Tie']]
        p=np.array([[.9,.05,.05]]*4)
        a=admission(rows,p,{'p_better':.5,'p_worse_max':1})
        self.assertEqual(a['accepted'],4)
        self.assertEqual(a['confirmed_precision_lower'],.25)
        self.assertEqual(a['possible_precision_upper'],.5)
        self.assertEqual(a['worst_case_utility_per_pair'],-.25)
        empty=admission(rows,p,None)
        self.assertIsNone(empty['confirmed_precision_lower'])

    def test_threshold_rejects_uncertain_inflated_precision(self):
        rows=[self.row('Better'),self.row('Uncertain')]
        z=np.array([[5.,0.,-1.],[5.,0.,-1.]])
        result=select_threshold(rows,z,1.,{'p_better_grid':[.5],'p_worse_max_grid':[1.]})
        self.assertIsNone(result['threshold'])
        self.assertEqual(result['status'],'NO_FEASIBLE_THRESHOLD')


if __name__=='__main__':unittest.main()
