"""Shared figure style for the camera-ready: 9pt Georgia at final print width, one palette."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

for f in (Path(__file__).resolve().parents[1] / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))

TEXTWIDTH = 5.5  # inches, NeurIPS single column
plt.rcParams.update({
    "font.family": "Georgia", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "pdf.fonttype": 42, "savefig.pad_inches": 0.02,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})
# Okabe-Ito based, colour-blind safe. C2 (headline condition) gets the boldest colour.
COND = {"C1": "#CC79A7", "C2": "#0072B2", "C3": "#009E73", "C4": "#E69F00", "C5": "#8C8C8C", "C6": "#D55E00"}
MARK = {"C1": "D", "C2": "o", "C3": "s", "C4": "v", "C5": "x", "C6": "^"}
SCEN = {"greenfield": "#2E86C1", "riverton": "#28B463"}
LABEL = {"C1": "C1 open", "C2": "C2 segregated + briefs", "C3": "C3 segregated, no brief",
         "C4": "C4 brief only", "C5": "C5 floor", "C6": "C6 solo generation"}


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}")


def save(fig, name):
    out = Path(__file__).resolve().parents[1] / "figures" / name
    fig.savefig(out, bbox_inches="tight"); print("wrote", out)
