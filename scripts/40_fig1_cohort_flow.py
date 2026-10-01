import matplotlib.pyplot as plt

import config as c

# Figure 1: flow of patients through the study (numbers as reported in the paper)

plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = 9

fig, ax = plt.subplots(figsize=(7.0, 7.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

box = dict(boxstyle="square,pad=0.6", facecolor="white", edgecolor="black", linewidth=0.8)
LW = 0.8


def node(x, y, text, ha="center"):
    return ax.text(x, y, text, ha=ha, va="center", bbox=box, linespacing=1.4)


def arrow(xy_from, xy_to, patch_from=None, patch_to=None, style="-|>"):
    ax.annotate("", xy=xy_to, xytext=xy_from, zorder=5,
                arrowprops=dict(arrowstyle=style, color="black", linewidth=LW, mutation_scale=10,
                                shrinkA=0, shrinkB=0, patchA=patch_from, patchB=patch_to))


def down(a, b):
    arrow(a.get_position(), b.get_position(), a.get_bbox_patch(), b.get_bbox_patch())


def side(x, y, excl):
    arrow((x, y), (excl.get_position()[0] + 5, y), None, excl.get_bbox_patch())


X = 32
XE = 59

n0 = node(X, 95, "Patients in MIMIC-IV v3.0\nn = 364,627")
e1 = node(XE, 84, "Excluded (n = 278,728)\nMeasurement criterion not met:\n≥3 determinations of each of the 17\nanalytes and ≥3 blood pressure readings", ha="left")
n1 = node(X, 73, "Eligible patients\nn = 85,899")
e2 = node(XE, 63, "Excluded (n = 5,601)\nNo hospital admission record", ha="left")
n2 = node(X, 53, "Patients with an\nadmission record\nn = 80,298")
e3 = node(XE, 43.5, "Excluded (n = 440)\n≥1 value outside plausibility limits", ha="left")
n3 = node(X, 32, "Analytical cohort\nn = 79,858\nHypertension: 49,655 (62.2%)\nControls: 30,203")
tr = node(17, 10, "Training set (75%)\nn = 59,893\nHypertension: 37,241\nControls: 22,652")
te = node(57, 10, "Test set (25%)\nn = 19,965\nHypertension: 12,414\nControls: 7,551")

down(n0, n1)
side(X, 84, e1)
down(n1, n2)
side(X, 63, e2)
down(n2, n3)
side(X, 43.5, e3)

arrow(n3.get_position(), (X, 21), n3.get_bbox_patch(), None, style="-")
ax.plot([17, 57], [21, 21], color="black", linewidth=LW, solid_capstyle="butt")
ax.text(X + 1, 23.5, "Stratified random split (patient level)", ha="left", va="center")
arrow((17, 21), tr.get_position(), None, tr.get_bbox_patch())
arrow((57, 21), te.get_position(), None, te.get_bbox_patch())

fig.savefig(c.FIGURES / "fig1_cohort_flow.svg", bbox_inches="tight")
