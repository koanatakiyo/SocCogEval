import sys; sys.path.insert(0, 'tools')
from dio import Dio, fit_heights
FS = 25.1
d = Dio('src/fig2_cartoon.drawio')
for c in d.texts():
    d.set_style(c, fontSize=FS)
titles = {'3': '(a) Gap', '5': '(b) Clones', '7': '(c) Interaction', '9': '(d) Probes', '11': '(e) Drift'}
for k, v in titles.items():
    t = d.cells[k]; g = d.geo(t)
    t.set('value', v); d.set_geo(t, x=g['x'] - 12, width=254, y=10, height=40)
for k in ['2', '4', '6', '8', '10']:
    d.cells[k].set('value', '')
V = {'50': 'No History', '52': 'Personas', '113': 'Confound', '631': 'Missing Clean\nEvidence',
     '632': 'Same Model,\nPrompt, Decoding', '633': 'Repeated\nDiscussion',
     '634': 'Off-Topic\nEvaluation', '635': 'Groups\nDiverge'}
for k, v in V.items():
    d.cells[k].set('value', v)
for k in ['631', '632', '633', '634', '635']:
    g = d.geo(d.cells[k]); d.set_geo(d.cells[k], y=548, height=70, x=g['x'] - 10, width=g['width'] + 20)
for k in ['504', '561', '622']:
    g = d.geo(d.cells[k]); d.set_geo(d.cells[k], x=g['x'] - 25, width=g['width'] + 50)
d.set_geo(d.cells['239'], y=50, height=34)
g = d.geo(d.cells['622']); d.set_geo(d.cells['622'], y=g['y'] + 10)
for k in ['631', '632', '633', '634', '635']:
    d.set_geo(d.cells[k], y=560)
fit_heights(d, FS)
skip = set(titles) | {'2', '4', '6', '8', '10', '239', '631', '632', '633', '634', '635'}
for c in d.root.iter('mxCell'):
    if c.get('parent') != '1' or c.get('id') in skip:
        continue
    g = d.geo(c)
    if g is None:
        continue
    dy = -14 if 80 <= g['y'] <= 560 else 0
    if g['x'] >= 975 and 80 <= g['y'] <= 560:
        dy -= 14                      # (e) column sits higher so its last label clears the tagline row
    if dy:
        d.set_geo(c, y=g['y'] + dy)
    for pt in c.iter('mxPoint'):
        if pt.get('y') is not None and 80 <= float(pt.get('y')) <= 560:
            pt.set('y', f"{float(pt.get('y')) - 14:g}")
for k in ['631', '632', '633', '634', '635']:
    d.set_geo(d.cells[k], y=518)
d.save('work/fig2_cartoon.drawio')
