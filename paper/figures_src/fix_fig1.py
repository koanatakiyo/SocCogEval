import sys; sys.path.insert(0, 'tools')
from dio import Dio
FS, DY1, DY2 = 43, 30, 85
MID = (DY1 + DY2) / 2
d = Dio('src/fig1.drawio')
special = {'4', '12', '30', '740', '155', '156', '469'}
def dy_for(x, y, w):
    if x + w / 2 < 165:          # left column (model icon, Clones) moves as one block
        return MID
    return DY2 if y >= 405 else DY1 if y >= 105 else 0
for c in d.root.iter('mxCell'):
    if c.get('parent') != '1' or c.get('id') in special:
        continue
    g = d.geo(c)
    if g is None:
        continue
    d.set_geo(c, y=g['y'] + dy_for(g['x'], g['y'], g['width']))
    for p in c.iter('mxPoint'):
        if p.get('y') is not None:
            x = float(p.get('x', 0)); y = float(p.get('y'))
            p.set('y', f"{y + dy_for(x, y, 0):g}")
for c in d.texts():
    d.set_style(c, fontSize=FS)
for k in ('4', '12'):
    g = d.geo(d.cells[k]); d.set_geo(d.cells[k], y=g['y'] + DY1, height=g['height'] + (DY2 - DY1))
d.set_geo(d.cells['155'], x=331, width=400, y=92, height=60)
d.set_geo(d.cells['156'], x=311, width=440, y=431 + DY2 - 62, height=60)
g = d.geo(d.cells['469']); d.set_geo(d.cells["469"], x=496, width=222, y=g['y'] + DY1 + 2, height=56)
d.set_style(d.cells['469'], fontStyle=0, align='left'); d.cells['469'].set('value', '30 rounds')
d.set_geo(d.cells["30"], x=-14, width=194, y=478 + MID + 8, height=60)
g = d.geo(d.cells['12']); cy = g['y'] + g['height'] / 2
gb = d.geo(d.cells['12']); cx = gb['x'] + gb['width'] + 34
d.set_geo(d.cells['740'], x=cx - 230, width=460, y=cy - 30, height=60)
d.save('work/fig1.drawio')
