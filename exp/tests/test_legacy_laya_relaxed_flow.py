import unittest
from legacy_laya_v1.laya_relaxed_policy import select_candidate
from legacy_laya_v1.laya_relaxed_flow import positive_memories


class RelaxedFlowTests(unittest.TestCase):
    def test_half_precision_is_not_half_probability(self):
        rows=[{'valid':True,'changed':True,'probabilities':[.28,.65,.07]},
              {'valid':True,'changed':True,'probabilities':[.32,.15,.53]}]
        self.assertEqual(select_candidate(rows,{'p_better':.275,'p_worse_max':.4}),0)

    def test_invalid_and_identical_do_not_get_accepted(self):
        rows=[{'valid':False,'changed':True,'probabilities':[.9,.05,.05]},
              {'valid':True,'changed':False,'probabilities':[.9,.05,.05]}]
        self.assertIsNone(select_candidate(rows,{'p_better':.275,'p_worse_max':.4}))
        self.assertIsNone(select_candidate(rows,{},'unfiltered'))

    def test_admission_uses_actual_before_and_ignores_acceptance(self):
        class Feedback:
            def primary(self,reference,text):return {'initial':10.,'accepted_previous':20.,'good_rejected':23.,'bad':19.}[text]
        trace={'source':'English input','initial':'initial','rounds':[{'before':'accepted_previous','accepted':False,
              'candidates':[{'text':'good_rejected','valid':True,'changed':True},
                            {'text':'good_rejected','valid':True,'changed':True},
                            {'text':'bad','valid':True,'changed':True}]}]}
        seen=set();mem=positive_memories(trace,'SECRET_REFERENCE',Feedback(),seen)
        self.assertEqual(len(mem),1)
        self.assertEqual(mem[0].state_before,'accepted_previous')
        self.assertEqual(mem[0].state_after,'good_rejected')
        self.assertEqual(mem[0].delta_offline,3.)
        self.assertNotIn('SECRET_REFERENCE',str(mem[0].to_dict()))
        self.assertEqual(positive_memories(trace,'SECRET_REFERENCE',Feedback(),seen),[])


if __name__=='__main__':unittest.main()
