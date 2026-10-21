"""Fig 4 re-laid out to the user's target sketch (2026-10-21), built from the user's 10:25 drawio icons.

Rows: shared initialization, C2 social interaction, C6 solo generation, C4 brief only.
Icons are moved/scaled; connectors are redrawn as draw.io edges; labels print at 8pt at full text width.
"""
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, 'tools')
from dio import Dio, delete_cell, scale_group

FS = float(sys.argv[1]) if len(sys.argv) > 1 else 27.5
d = Dio('src/fig4_v3_user.drawio')

# ---------- remove what the new layout does not use ----------
DROP = ['42', '43', '44', '45', '46', '47', '153', '154', '155', '164', '171', '196', '203', '204',
        '206', '207', '208', '209', '210', '211', '212', '213', '232', '233',
        '274', '294', '358', '359', '360', '361', '376', '381',
        '442', '443', '455', '456', '480', '481', '547', '548', '558', '559', '565', '566',
        '575', '576', '600', '601', '623', '624', '625', '626', '627', '628', '648', '649', '650',
        '215', '504', '631']
for k in DROP:
    if k in d.cells:
        delete_cell(d, k)
d.cells = {c.get('id'): c for c in d.root.iter('mxCell')}
root_el = next(d.root.iter('root'))


def place(ids, ox, oy, nx, ny, s=1.0):
    """Move top-level cells so (ox, oy) maps to (nx, ny), scaling positions and sizes by s."""
    for k in ids:
        c = d.cells[k]; g = d.geo(c)
        st = (c.get('style') or '')
        if st.startswith('group'):
            scale_group(d, k, s)
        else:
            d.set_geo(c, width=g['width'] * s, height=g['height'] * s)
        d.set_geo(c, x=nx + (g['x'] - ox) * s, y=ny + (g['y'] - oy) * s)


def text(k, value, cx, y, w, lines=1, bold=False, align='center', size=FS):
    c = d.cells[k]
    c.set('value', value)
    d.set_style(c, fontSize=size, fontStyle=1 if bold else 0, align=align, verticalAlign='middle',
                spacing=0, whiteSpace='wrap')
    x = cx - w / 2 if align == 'center' else cx
    d.set_geo(c, x=x, y=y, width=w, height=lines * size * 1.22)


_n = [0]
def new_text(value, cx, y, w, lines=1, bold=False):
    _n[0] += 1
    c = ET.SubElement(root_el, 'mxCell', id=f'cr_t{_n[0]}', value=value, vertex='1', parent='1',
                      style='text;html=0;fontFamily=Georgia;fontColor=#2B2B2B;')
    ET.SubElement(c, 'mxGeometry', x='0', y='0', width='10', height='10', **{'as': 'geometry'})
    d.cells[c.get('id')] = c
    text(c.get('id'), value, cx, y, w, lines, bold)


def edge(points, style):
    _n[0] += 1
    c = ET.SubElement(root_el, 'mxCell', id=f'cr_e{_n[0]}', edge='1', parent='1', style=style)
    g = ET.SubElement(c, 'mxGeometry', relative='1', **{'as': 'geometry'})
    (x0, y0), (x1, y1) = points[0], points[-1]
    ET.SubElement(g, 'mxPoint', x=f'{x0:g}', y=f'{y0:g}', **{'as': 'sourcePoint'})
    ET.SubElement(g, 'mxPoint', x=f'{x1:g}', y=f'{y1:g}', **{'as': 'targetPoint'})
    if len(points) > 2:
        arr = ET.SubElement(g, 'Array', **{'as': 'points'})
        for x, y in points[1:-1]:
            ET.SubElement(arr, 'mxPoint', x=f'{x:g}', y=f'{y:g}')


GRAY = '#8C8C9A'
ARROW = f'endArrow=block;endFill=1;endSize=7;strokeColor={GRAY};strokeWidth=3;html=1;'
CURVE = f'curved=1;endArrow=block;endFill=1;endSize=7;strokeColor={GRAY};strokeWidth=3;html=1;'
DOT_O = 'curved=1;dashed=1;dashPattern=1 3;strokeColor=#EDB596;strokeWidth=4;endArrow=none;html=1;'
DOT_B = 'dashed=1;dashPattern=1 3;strokeColor=#A8C1E3;strokeWidth=4;endArrow=block;endFill=1;endSize=7;html=1;'
DASH = 'dashed=1;dashPattern=8 6;strokeColor=#B8B8C2;strokeWidth=3;endArrow=none;html=1;'
DASH_A = 'dashed=1;dashPattern=8 6;strokeColor=#B8B8C2;strokeWidth=3;endArrow=block;endFill=1;endSize=7;html=1;'
LH = FS * 1.22

# ---------- row 1: shared initialization ----------
text('6', 'Shared Initialization', 30, 6, 420, bold=True, align='left')
place(['18'], 102, 78, 40, 58, 1.0)                                   # source (78x97)
text('41', '1 Source', 79, 160, 150)
edge([(128, 108), (178, 108)], ARROW)
agents = ['48', '61', '74', '87', '100', '113', '126', '139']
for i, k in enumerate(agents):
    g = d.geo(d.cells[k])
    place([k], g['x'], g['y'], 192 + (i % 4) * 46, 62 + (i // 4) * 58, 1.35)
text('152', '8 Agents', 280, 160, 160)
edge([(375, 108), (405, 108), (420, 64), (440, 62)], CURVE)
edge([(375, 108), (405, 108), (420, 152), (440, 154)], CURVE)
DOCT = 1.15
for k, (x, y) in {'156': (448, 20), '172': (528, 20), '180': (448, 112), '188': (528, 112)}.items():
    place([k], d.geo(d.cells[k])['x'], d.geo(d.cells[k])['y'], x, y, DOCT)
for lab, doc, letter in [('163', '156', 'A'), ('179', '172', 'B'), ('187', '180', 'A'), ('195', '188', 'B')]:
    g = d.geo(d.cells[doc])
    text(lab, f'Brief\n{letter}', g['x'] + g['width'] / 2, g['y'] + g['height'] * 0.22, g['width'] - 4, lines=2)
text('205', 'Matched\nBriefs', 685, 70, 150, lines=2)
edge([(690, 128), (690, 200), (600, 240), (560, 300)], DOT_O)
# upper right: the same post-probes close every condition (matched briefs -> probes column)
DOT_OA = 'curved=1;dashed=1;dashPattern=1 3;strokeColor=#EDB596;strokeWidth=4;endArrow=block;endFill=1;endSize=7;html=1;'
edge([(765, 96), (1180, 92), (1305, 120), (1310, 262)], DOT_OA)
new_text('Same post-probes\nfor every condition', 1010, 112, 300, lines=2)

# ---------- row 2: C2 social interaction ----------
R2 = 250
text('214', 'C2 Social\nInteraction', 20, R2 + 25, 180, lines=2, bold=True, align='left')
CD = 1.3
def docs(a, b, la, lb, y):
    ga, gb = d.geo(d.cells[a]), d.geo(d.cells[b])
    place([a], ga['x'], ga['y'], 195, y, CD)
    place([b], gb['x'], gb['y'], 266, y + 26, CD)
    for lab, doc, letter in [(la, a, 'A'), (lb, b, 'B')]:
        g = d.geo(d.cells[doc])
        text(lab, f'Brief\n{letter}', g['x'] + g['width'] / 2, g['y'] + g['height'] * 0.22, g['width'] - 4, lines=2)
docs('216', '224', '223', '231', R2 + 5)
edge([(349, R2 + 62), (560, R2 + 62)], DOT_B)
PS = 1.2
pair1 = ['234', '254', '354', '355', '356', '357', '366', '371']
pair2 = ['314', '334', '362', '363', '364', '365', '386', '391']
place(pair1, 386, 266, 575, R2 - 10, PS)
place(pair2, 791, 266, 775, R2 - 10, PS)
text('441', '4 Pairs & 6 Turns', 765, R2 + 120, 320)
place(['444'], 976, 287, 975, R2 + 30, 0.95)                         # dialogue record
text('454', 'Dialogue-\nrecord', 1006, R2 - 48, 160, lines=2)
edge([(1045, R2 + 66), (1083, R2 + 66)], ARROW)
place(['457'], 1090, 294, 1090, R2 + 20, 1.0)                         # SRM
text('479', 'SRM\nExtraction', 1143, R2 - 48, 170, lines=2)
edge([(1150, R2 + 112), (1120, R2 + 150), (1040, R2 + 150), (1010, R2 + 105)], CURVE)
text('629', '30 Rounds', 1080, R2 + 150, 170)
edge([(1200, R2 + 66), (1262, R2 + 66)], ARROW)
place(['482'], 1277, 284, 1272, R2 + 20, 1.0)
text('502', 'Same\nPost-Probes', 1315, R2 + 112, 170, lines=2)

SEP1 = R2 + 200
d.set_geo(d.cells['3'], y=SEP1)

# ---------- row 3: C6 solo generation ----------
R6 = SEP1 + 62
text('503', 'C6 Solo\nGeneration', 20, R6 + 30, 180, lines=2, bold=True, align='left')
docs('505', '513', '512', '520', R6 + 8)
text('521', '×8\nAgents', 395, R6 + 30, 100, lines=2, bold=True)
place(['522', '542', '543', '544', '545'], 433, 440, 455, R6 + 2, 1.05)
text('545', '?', d.geo(d.cells['545'])['x'] + 13, d.geo(d.cells['545'])['y'], 26, size=22)
text('546', 'No Partner', 490, R6 + 100, 170)
IC = R6 + 25
edge([(540, IC + 40), (572, IC + 40)], ARROW)
place(['549'], 593, 441, 580, IC, 1.0)
text('557', 'Initial\nReasoning', 616, IC + 84, 150, lines=2)
edge([(660, IC + 40), (740, IC + 40)], ARROW)
place(['560'], 747, 441, 749, IC, 1.0)
text('564', 'Reconsideration', 785, IC + 84, 220)
edge([(829, IC + 40), (930, IC + 40)], ARROW)
place(['567'], 901, 441, 939, IC, 1.0)
text('574', 'Allocation\nReflection', 975, IC + 84, 160, lines=2)
edge([(1019, IC + 40), (1083, IC + 40)], ARROW)
place(['577'], 1090, 456, 1090, IC - 5, 1.0)
text('599', 'SRM\nExtraction', 1143, IC + 90, 170, lines=2)
edge([(1170, IC - 6), (1185, IC - 30), (1105, IC - 30), (1112, IC - 4)], CURVE)
new_text('30 Rounds', 1145, IC - 30 - LH - 2, 170)
edge([(1200, IC + 40), (1262, IC + 40)], ARROW)
place(['602'], 1277, 447, 1272, IC - 5, 1.0)
text('622', 'Same\nPost-Probes', 1315, IC + 88, 170, lines=2)

SEP2 = IC + 90 + 2 * LH + 14
d.set_geo(d.cells['4'], y=SEP2)

# ---------- row 4: C4 brief only ----------
R4 = SEP2 + 22
text('630', 'C4\nBrief Only', 20, R4 + 22, 180, lines=2, bold=True, align='left')
docs('632', '640', '639', '647', R4)
MID = R4 + 58
place(['651', '652', '664', '665'], 700, 592, 690, R4 + 5, 1.0)
edge([(330, MID), (680, MID)], DASH)
edge([(840, MID), (1262, MID)], DASH_A)
text('664', 'zZ', d.geo(d.cells['664'])['x'] + 17, d.geo(d.cells['664'])['y'], 40, size=24)
text('667', 'No Activity', 755, R4 + 88, 200)
place(['668'], 1277, 594, 1272, R4 + 5, 1.0)
text('688', 'Same\nPost-Probes', 1315, R4 + 97, 170, lines=2)

d.save('work/fig4_cartoon.drawio')
print('bottom', round(R4 + 97 + 2 * LH))
