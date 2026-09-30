import copy
import unittest
from acceptance_data import choose_next, valid_candidate, unique_rows, candidate_temperature

class CollectionInvariants(unittest.TestCase):
    def test_mixed_sampling_is_training_only(self):
        p = {'train_candidate_temperatures': [.1,.1,.7,.7]}
        self.assertEqual([candidate_temperature('train', s, p) for s in range(4)], [.1,.1,.7,.7])
        for fold in ['development','temperature_calibration','threshold_calibration','test']:
            self.assertEqual([candidate_temperature(fold,s,p) for s in range(4)], [.1]*4)

    def test_feedback_cannot_change_state_selection(self):
        cs = [{'text': t, 'finish_reason': 'stop'} for t in ['好的修改。','很差修改。','原来内容。','别的修改。']]
        a = choose_next('source_a', '原来内容。', cs)
        altered = copy.deepcopy(cs)
        for i, c in enumerate(altered):
            c['delta'] = 100 if i != a else -100
            c['accepted'] = i != a
        self.assertEqual(a, choose_next('source_a', '原来内容。', altered))
        selected = {choose_next(str(i), '原来内容。', cs) for i in range(200)}
        self.assertEqual(selected, {0,1,2,3})

    def test_invalid_excluded_but_identical_retained(self):
        cs = [{'text': '原来内容。', 'finish_reason': 'stop'},
              {'text': '未完成。', 'finish_reason': 'length'},
              {'text': '', 'finish_reason': 'stop'}]
        self.assertEqual(choose_next('x', '原来内容。', cs), 0)
        self.assertFalse(valid_candidate('短句。', {'text':'长'*100, 'finish_reason':'stop'})[0])
        self.assertIsNone(choose_next('x', '原来内容。', cs[1:]))

    def test_dedup_preserves_different_before_state_and_sources(self):
        g = {'text':'新结果。', 'finish_reason':'stop', 'seed':1}
        r = {'sample_id':'a','source':'source a', 'states':[
             {'index':0,'before':'旧结果。','candidates':[g,g]},
             {'index':1,'before':'别的结果。','candidates':[g]}]}
        s = copy.deepcopy(r); s['source']='source b'; s['sample_id']='b'
        rows = list(unique_rows([r,s]))
        self.assertEqual(len(rows), 4)

if __name__ == '__main__':
    unittest.main()
