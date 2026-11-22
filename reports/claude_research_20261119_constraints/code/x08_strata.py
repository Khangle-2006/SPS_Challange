"""x08 - group analyses for C2 (skin), C6 (age), plus sex, organiser group and real lighting labels - BTC dev (218 videos / 158 people).
Group gate of PREREG.md: every group with >= 20 people within +-2 bpm of the overall person-first MAE, and a permutation test of the
group labels between people (10,000, statistic = max - min group MAE over groups with >= 20 people) not rejecting at 5%.
Age and sex come from the organisers' workbook: only columns A (id), D (sex), E (age), F (skin) of the 218 DEV rows are read
(`fetch` mode); no name, no holdout row, and the workbook copy is deleted after the three columns are written.
Usage: $PY code/x08_strata.py fetch            -> out/btc_dev_demographics.csv
       $PY code/x08_strata.py score [predfile=out/x01_btc_pred.npz] [keys ...]"""
import sys, csv, json, zipfile, subprocess, xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints')
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'; REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
def read_cells(path, wanted):
    """wanted {sheet: {row_number: set(column letters)}} -> {(sheet, row, col): text}; nothing else is kept."""
    z = zipfile.ZipFile(path); rels = {x.attrib['Id']: x.attrib['Target'] for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
    wb = ET.fromstring(z.read('xl/workbook.xml')); sheets = {s.attrib['name'].strip(): rels[s.attrib[REL]] for s in wb.find('{%s}sheets' % NS)}
    ss = [''.join(t.text or '' for t in si.iter('{%s}t' % NS)) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))] if 'xl/sharedStrings.xml' in z.namelist() else []
    out = {}
    for name, rows in wanted.items():
        tg = sheets[name]; tg = tg.lstrip('/') if tg.startswith('/') else 'xl/' + tg
        for row in ET.fromstring(z.read(tg)).iter('{%s}row' % NS):
            r = int(row.attrib['r'])
            if r not in rows: continue
            for c in row:
                col = ''.join(ch for ch in c.attrib['r'] if ch.isalpha())
                if col not in rows[r]: continue
                v = c.find('{%s}v' % NS); t = c.attrib.get('t')
                out[(name, r, col)] = ss[int(v.text)] if (t == 's' and v is not None) else (''.join(x.text or '' for x in c.iter('{%s}t' % NS)) if t == 'inlineStr' else (None if v is None else v.text))
    return out
def btc_jobs():
    import ast
    jobs = [j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] == 'BTC']
    for j in jobs:
        if isinstance(j['metadata'], str): j['metadata'] = ast.literal_eval(j['metadata'])
    return jobs
def fetch():
    url = json.loads((R.parent / 'causal_audit_20261113/WORKBOOK_FETCH.json').read_text())['url']; tmp = W / 'out/tmp_workbook.xlsx'
    subprocess.check_call(['curl', '-sSL', '--max-time', '120', '-o', str(tmp), url]); assert zipfile.is_zipfile(tmp), 'download is not an xlsx'
    jobs = btc_jobs(); wanted = defaultdict(dict)
    for j in jobs: wanted[j['metadata']['group']][int(j['metadata']['sheet_row'])] = {'A', 'D', 'E', 'F'}
    cells = read_cells(tmp, wanted); rows = []; bad = 0
    for j in jobs:
        g, r = j['metadata']['group'], int(j['metadata']['sheet_row']); rid = (cells.get((g, r, 'A')) or '').strip()
        if rid != j['recording'].strip(): bad += 1; continue
        age = cells.get((g, r, 'E')); sex = (cells.get((g, r, 'D')) or '').strip(); skin = (cells.get((g, r, 'F')) or '').strip().lower()
        rows.append(dict(recording=j['recording'], person=j['person'], age='' if age in (None, '') else float(age), sex=sex, skin_sheet=skin, skin_manifest=j['metadata']['skin']))
    with (W / 'out/btc_dev_demographics.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    tmp.unlink(); ages = np.array([r['age'] for r in rows if r['age'] != ''])
    print('dev rows matched', len(rows), 'id mismatches', bad, '| age present', len(ages), 'range', ages.min(), ages.max(), '| sex', Counter(r['sex'] for r in rows), '| skin differs from manifest in', sum(r['skin_sheet'] != r['skin_manifest'].lower() for r in rows), 'rows | workbook copy deleted')
def pmean(err, per, sel):
    d = defaultdict(list)
    for e, p, s in zip(err, per, sel):
        if s: d[p].append(e)
    k = sorted(d); return k, np.array([np.mean(d[p]) for p in k])
def group_report(name, err, per, lab_of_person, min_n=20, nperm=10000):
    """person-level attribute: MAE per group with CI, gate, permutation p."""
    ppl, v = pmean(err, per, np.ones(len(err), bool)); g = np.array([lab_of_person.get(p, 'unknown') for p in ppl]); overall = v.mean(); rng = np.random.default_rng(0); out = []
    big = [k for k, c in Counter(g).items() if c >= min_n and k != 'unknown']
    for k in sorted(set(g)):
        x = v[g == k]; ci = np.quantile(x[rng.integers(0, len(x), (10000, len(x)))].mean(1), [.025, .975]) if len(x) > 1 else [np.nan, np.nan]
        out.append(f"{k}: n={len(x)} MAE {x.mean():.2f} [{ci[0]:.2f};{ci[1]:.2f}] ({x.mean() - overall:+.2f})")
    line = f"  {name:16s} overall {overall:.3f} | " + ' | '.join(out)
    if len(big) >= 2:
        stat = lambda gg: np.ptp([v[gg == k].mean() for k in big]); obs = stat(g); m = np.isin(g, big); gb = g.copy(); cnt = 0
        for _ in range(nperm):
            gb[m] = rng.permutation(g[m]); cnt += stat(gb) >= obs - 1e-12
        worst = max(abs(v[g == k].mean() - overall) for k in big); p = (cnt + 1) / (nperm + 1)
        line += f" || spread {obs:.2f}, worst deviation {worst:.2f}, permutation p={p:.3f} -> gate {'PASS' if (worst <= 2. and p >= .05) else 'FAIL'}"
    else: line += ' || fewer than two groups with >= 20 people: no gate'
    print(line, flush=True)
if __name__ == '__main__':
    if sys.argv[1] == 'fetch': fetch(); sys.exit()
    z = np.load(sys.argv[2] if len(sys.argv) > 2 else W / 'out/x01_btc_pred.npz'); rec, lab, per = z['rec'], z['lab'], z['per']
    keys = sys.argv[3:] or ['native_refit'] + [k for k in z.files if k.endswith('|refit') and any(k.startswith(c) for c in ('c1_g25p|', 'c4_vga|std|seen', 'c7_blue|oa', 'c6_us1', 'c5_tuned', 'c5_face'))]
    INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}; rec = np.array([INV.get(r, r) for r in rec])   # export alias -> real name
    J = {j['recording']: j for j in btc_jobs()}; demo = {}
    if (W / 'out/btc_dev_demographics.csv').exists(): demo = {r['recording']: r for r in csv.DictReader((W / 'out/btc_dev_demographics.csv').open())}
    first = {}
    for r, p in zip(rec, per): first.setdefault(p, r)
    skin = {p: J[r]['metadata']['skin'].lower() for p, r in first.items()}; grp = {p: J[r]['metadata']['group'] for p, r in first.items()}
    def ageband(a): return 'unknown' if a in ('', None) else ('<40' if float(a) < 40 else ('40-59' if float(a) < 60 else '>=60'))
    age = {p: ageband(demo[r]['age']) if r in demo else 'unknown' for p, r in first.items()}; age70 = {p: ('unknown' if r not in demo or demo[r]['age'] == '' else ('>=70' if float(demo[r]['age']) >= 70 else '<70')) for p, r in first.items()}
    sex = {p: (demo[r]['sex'].lower() or 'unknown') if r in demo else 'unknown' for p, r in first.items()}
    light = np.array([J[r]['metadata']['lighting'] for r in rec])
    print('people', len(first), '| skin', Counter(skin.values()), '| age', Counter(age.values()), '| >=70', Counter(age70.values()), '| sex', Counter(sex.values()), '| lighting (videos)', Counter(light))
    for k in keys:
        okv = np.isfinite(z[k]); err = np.abs(z[k] - lab)[okv]; pk = per[okv]; print(f'\n=== {k} ({int(okv.sum())} videos)')
        group_report('skin (C2)', err, pk, skin); group_report('age (C6)', err, pk, age); group_report('age 70+ (C6)', err, pk, age70); group_report('sex', err, pk, sex); group_report('organiser group', err, pk, grp)
        lp = {}
        for p, l in zip(pk, light[okv]):
            if l in ('Bright', 'Med', 'Dim'): lp.setdefault(p, str(l))
        m = np.array([p in lp for p in pk]); group_report('lighting (real)', err[m], pk[m], lp)       # one lighting level per person (their three Task videos)
