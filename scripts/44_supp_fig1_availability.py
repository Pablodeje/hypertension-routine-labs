import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as c

# Supplementary Figure 1: availability of the 17 analytes among patients with laboratory data

ANALYTES = [("Sodium", "Sodium"), ("Potassium", "Potassium"), ("Chloride", "Chloride"),
            ("Bicarbonate", "Bicarbonate"), ("Anion GAP", "Anion gap"), ("Urea Nitrogen", "Urea nitrogen"),
            ("Creatinine", "Creatinine"), ("Glucose", "Glucose"), ("Hemoglobin", "Haemoglobin"),
            ("Hematocrit", "Haematocrit"), ("Red Blood Cells", "RBC"),
            ("MCV", "MCV"), ("MCH", "MCH"),
            ("MCHC", "MCHC"), ("RDW", "RDW"),
            ("White Blood Cells", "WBC"), ("Platelet Count", "Platelet count")]
COLOR = "#0072B2"

plt.rcParams.update({"font.family": "Arial", "font.size": 9})

m = pd.read_parquet(c.AVAILABILITY)[[a for a, _ in ANALYTES]]   # True = >=3 determinations
m.columns = [label for _, label in ANALYTES]

available = m.mean() * 100
complete = m.all(axis=1).sum()
gain = pd.Series({col: m.drop(columns=col).all(axis=1).sum() - complete for col in m.columns})
correlation = m.astype(int).corr()

fig = plt.figure(figsize=(7.2, 8.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.3, wspace=0.08)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, :])

y = np.arange(len(m.columns))[::-1]
axA.barh(y, available.values, color=COLOR, height=0.7)
axA.axvline(50, color="grey", linestyle="--", linewidth=0.8, zorder=3)
axA.set_yticks(y, m.columns)
axA.set_xlim(0, 100)
axA.set_xlabel("Patients with ≥3 determinations (%)")

axB.barh(y, gain.values, color=COLOR, height=0.7)
axB.set_yticks(y, [])
axB.set_xlabel("Additional complete patients\nif the analyte were not required (n)")

for ax in (axA, axB):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

im = axC.imshow(correlation.values, cmap="cividis", vmin=0.8, vmax=1)
axC.set_xticks(range(len(m.columns)), m.columns, rotation=90)
axC.set_yticks(range(len(m.columns)), m.columns)
cb = fig.colorbar(im, ax=axC, fraction=0.04, pad=0.02)
cb.set_label("Correlation of availability")

# Centre block C (labels + map + colour bar) under A+B
fig.canvas.draw()
inv = fig.transFigure.inverted()
ab = [ax.get_tightbbox().transformed(inv) for ax in (axA, axB)]
centre_ab = (min(b.x0 for b in ab) + max(b.x1 for b in ab)) / 2
bc = [a.get_tightbbox().transformed(inv) for a in (axC, cb.ax)]
dx = centre_ab - (min(b.x0 for b in bc) + max(b.x1 for b in bc)) / 2
for a in (axC, cb.ax):
    p = a.get_position()
    a.set_position([p.x0 + dx, p.y0, p.width, p.height])

# Letters aligned with the left edge of each panel (labels included)
fig.canvas.draw()
for ax, letter in ((axA, "A"), (axB, "B"), (axC, "C")):
    x0 = ax.get_tightbbox().transformed(fig.transFigure.inverted()).x0
    y1 = ax.get_position().y1
    fig.text(x0, y1 + 0.01, letter, fontsize=11, fontweight="bold", va="bottom")

fig.savefig(c.FIGURES / "supp_fig1_availability.svg", bbox_inches="tight")