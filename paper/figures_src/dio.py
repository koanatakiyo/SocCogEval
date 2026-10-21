"""Small helper to read and edit plain-XML draw.io files (text cells, fonts, geometry)."""
import re, xml.etree.ElementTree as ET

class Dio:
    def __init__(self, path):
        self.path = path
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self.cells = {c.get('id'): c for c in self.root.iter('mxCell')}

    def texts(self):
        out = []
        for c in self.root.iter('mxCell'):
            st = c.get('style') or ''
            if c.get('value') and st.startswith('text'):
                out.append(c)
        return out

    def geo(self, c):
        g = c.find('mxGeometry')
        return {k: float(g.get(k, 0)) for k in ('x', 'y', 'width', 'height')} if g is not None else None

    def set_geo(self, c, **kw):
        g = c.find('mxGeometry')
        for k, v in kw.items():
            g.set(k, f'{v:g}')

    def abs_xy(self, c):
        x = y = 0.0
        while c is not None and c.get('id') not in ('0', '1'):
            g = c.find('mxGeometry')
            if g is not None and g.get('relative') != '1':
                x += float(g.get('x', 0)); y += float(g.get('y', 0))
            c = self.cells.get(c.get('parent'))
        return x, y

    def style(self, c):
        return dict(kv.split('=', 1) if '=' in kv else (kv, '') for kv in (c.get('style') or '').split(';') if kv)

    def set_style(self, c, **kw):
        st = self.style(c); st.update({k: str(v) for k, v in kw.items()})
        c.set('style', ';'.join(f'{k}={v}' if v != '' else k for k, v in st.items()) + ';')

    def save(self, path=None):
        self.tree.write(path or self.path, encoding='utf-8', xml_declaration=False)

def fit_heights(d, fs, lead=1.25):
    """Grow every text cell to hold its lines at font size fs, keeping the vertical center."""
    for c in d.texts():
        n = c.get('value').count('\n') + 1
        need = n * fs * lead
        g = d.geo(c)
        if g['height'] < need:
            d.set_geo(c, y=g['y'] - (need - g['height']) / 2, height=need)

def descendants(d, cid):
    out, stack = [], [cid]
    while stack:
        p = stack.pop()
        for c in d.root.iter('mxCell'):
            if c.get('parent') == p:
                out.append(c); stack.append(c.get('id'))
    return out

def delete_cell(d, cid):
    parent = {c: p for p in d.root.iter() for c in p}
    for c in descendants(d, cid) + [d.cells[cid]]:
        parent[c].remove(c)

def scale_group(d, cid, s):
    """Scale a group's own size and every descendant's geometry by s (positions inside the group too)."""
    for c in [d.cells[cid]] + descendants(d, cid):
        g = c.find('mxGeometry')
        if g is None:
            continue
        own = c.get('id') == cid
        for k in (('width', 'height') if own else ('x', 'y', 'width', 'height')):
            if g.get(k) is not None:
                g.set(k, f"{float(g.get(k)) * s:g}")
        if not own:
            for p in g.iter('mxPoint'):
                for k in ('x', 'y'):
                    if p.get(k) is not None:
                        p.set(k, f"{float(p.get(k)) * s:g}")
        st = d.style(c)
        if 'strokeWidth' in st and not own:
            d.set_style(c, strokeWidth=f"{float(st['strokeWidth']) * (1 + (s - 1) * 0.5):g}")
