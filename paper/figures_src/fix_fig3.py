import sys; sys.path.insert(0, 'tools')
from dio import Dio
FS = 29.9
d = Dio('src/fig3.drawio')
for c in d.texts():
    d.set_style(c, fontSize=FS)
V = {  # id: new value (Title Case, paper naming)
 '14': 'Inputs', '15': 'Conditions', '16': 'Lifecycle', '17': 'Gates',
 '18': 'Clones', '157': 'Briefs', '176': 'Topology', '287': 'Open', '288': 'Segregated',
 '657': 'Interaction', '658': 'No Interaction',
 '829': 'Build', '830': 'Talk', '831': 'Probe', '832': 'Persist',
 '833': 'Build', '834': 'No Talk', '835': 'Probe', '836': 'Persist',
 '837': 'SocCogEval', '908': 'Robust', '912': 'Diverge', '916': 'Predict', '920': 'Transfer',
 '963': 'Verdict', '972': 'SCT-positive 2/2',
}
for k, v in V.items():
    d.cells[k].set('value', v)
G = {  # id: geometry overrides (absolute == relative for parent=1)
 '17': dict(y=48, height=48), '288': dict(x=180, width=170), '287': dict(x=40, width=150),
 '832': dict(x=1066, width=136), '836': dict(x=1066, width=136), '834': dict(x=836, width=136),
 '972': dict(x=1219, width=290),
}
for k, g in G.items():
    d.set_geo(d.cells[k], **g)
d.save('work/fig3.drawio')
