import unittest,json,copy,hashlib
from pathlib import Path
from build_split import assign
HERE=Path(__file__).resolve().parent
class Contract(unittest.TestCase):
    def test_people_split_and_no_holdout_manifest(self):
        s=json.loads((HERE/'btc_split.json').read_text(encoding='utf-8'))
        m=json.loads((HERE/'dev_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(m['split_sha256'],hashlib.sha256((HERE/'btc_split.json').read_bytes()).hexdigest())
        for r in m['videos']+m['missing']:
            self.assertEqual(s['people'][r['person']]['partition'],'dev')
            self.assertIn(r['recording'],s['people'][r['person']]['recordings'])
        self.assertEqual(len({r['person'] for r in m['videos']}),161)
        self.assertEqual(sum(p['partition']=='holdout' for p in s['people'].values()),41)
    def test_assignment_independent_of_input_order(self):
        people={f'S_N{i}':dict(group='Normal',skins={'dark' if i%3 else 'fair'}) for i in range(31)}
        x=copy.deepcopy(people);y=copy.deepcopy(dict(reversed(list(people.items()))))
        assign(x);assign(y);self.assertEqual(x,y)
    def test_whole_person_grouping(self):
        s=json.loads((HERE/'btc_split.json').read_text(encoding='utf-8'))
        records=[r for p in s['people'].values() for r in p['recordings']]
        self.assertEqual(len(records),len(set(records)))
        self.assertEqual({p['fold'] for p in s['people'].values() if p['partition']=='dev'},set(range(5)))
if __name__=='__main__':unittest.main()
