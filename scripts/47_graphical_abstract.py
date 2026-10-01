import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

import config as c

# Graphical abstract (numbers as reported in the paper)

# ---------------- Settings ----------------
SVG = c.FIGURES / "graphical_abstract.svg"    # editable
TIFF = c.FIGURES / "graphical_abstract.tif"   # for submission (format requested by the template)
WIDTH, HEIGHT = 18.0, 12.5         # cm (width x height), EHJ-DH instructions
CHARS_PER_LINE = 30               # text wrapping in the boxes (raise if there is room left in Arial)

# Headers in the colours of the journal template; graphic in Okabe-Ito, as in the figures
HEADER = {"Key question": "#5FA2CE", "Key findings": "#FC7D0B", "Take-home message": "#A3ACB9"}
BLUE, VERMILLION, GREY = "#0072B2", "#D55E00", "#7F7F7F"
GROUP = {"Renal function": "#D55E00", "Glucose": "#E69F00", "Electrolytes": "#0072B2",
         "White cells and platelets": "#009E73", "Red cells": "#CC79A7"}

TEXTS = {
    "Key question": "Hypertension is diagnosed from blood pressure alone. Do routine blood tests, "
                    "already collected during ordinary care, add information on hypertension "
                    "status beyond blood pressure?",
    "Key findings": "In 79,858 adults, adding routine laboratory predictors to blood pressure raised "
                    "discrimination from 0.822 to 0.881–0.885, with consistent gains across five "
                    "algorithms. Thirty selected predictors retained 93% of the gain.",
    "Take-home message": "Routine blood tests carry information on hypertension beyond blood pressure, "
                         "concentrated in a few analytes. Their value for screening requires validation "
                         "with treatment data and in screening populations.",
}
# Results (as reported in the paper; best algorithm, CatBoost)
AUROC = [("Blood pressure", 0.822, GREY), ("+ 30 (lasso)", 0.881, BLUE),
         ("+ 70 (SHAP)", 0.884, BLUE), ("+ all 102", 0.885, BLUE)]
ANALYTES_GA = [("Urea nitrogen", 16.8, "Renal function"), ("Glucose", 14.2, "Glucose"),
            ("Chloride", 9.9, "Electrolytes"), ("Creatinine", 9.0, "Renal function"),
            ("Potassium", 8.8, "Electrolytes"), ("White cells", 7.3, "White cells and platelets"),
            ("Red cell width", 5.8, "Red cells")]

plt.rcParams.update({"font.family": "Arial", "font.size": 10})
fig = plt.figure(figsize=(WIDTH / 2.54, HEIGHT / 2.54))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.axis("off")

# ---------------- Top boxes (template structure) ----------------
M, SEP, W_BOX = 0.4, 0.6, (WIDTH - 2 * 0.4 - 2 * 0.6) / 3
Y_BOX, H_BOX, H_HEAD = 7.55, 4.6, 0.62
for i, (title, text) in enumerate(TEXTS.items()):
    x = M + i * (W_BOX + SEP)
    ax.add_patch(Rectangle((x, Y_BOX), W_BOX, H_BOX, fill=False, edgecolor="#BFBFBF", linewidth=0.8))
    ax.add_patch(Rectangle((x, Y_BOX + H_BOX - H_HEAD), W_BOX, H_HEAD, color=HEADER[title], linewidth=0))
    ax.text(x + 0.15, Y_BOX + H_BOX - H_HEAD / 2, title, color="white", fontsize=11, fontweight="bold", va="center")
    ax.text(x + 0.15, Y_BOX + H_BOX - H_HEAD - 0.15, textwrap.fill(text, CHARS_PER_LINE), fontsize=10,
            va="top", linespacing=1.3)
    if i < 2:
        ax.add_patch(FancyArrowPatch((x + W_BOX + 0.08, Y_BOX + H_BOX / 2), (x + W_BOX + SEP - 0.08, Y_BOX + H_BOX / 2),
                                     arrowstyle="-|>", mutation_scale=14, color="black", linewidth=1.5))

# ---------------- Bottom graphic: data -> comparison -> findings ----------------
ax.add_patch(Rectangle((M, 0.35), WIDTH - 2 * M, 6.8, fill=False, edgecolor="#BFBFBF", linewidth=0.8))


def tube(x, y, h=1.1, w=0.38):
    # blood tube: body, blood and cap
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc="white", ec="#404040", lw=1))
    ax.add_patch(FancyBboxPatch((x + 0.04, y + 0.04), w - 0.08, h * 0.55, boxstyle="round,pad=0,rounding_size=0.14",
                                fc="#C0392B", ec="none"))
    ax.add_patch(Rectangle((x - 0.04, y + h - 0.05), w + 0.08, 0.22, fc="#5FA2CE", ec="#404040", lw=1))


def cuff(x, y):
    # sphygmomanometer: gauge and cuff
    ax.add_patch(Circle((x + 0.35, y + 0.75), 0.33, fc="white", ec="#404040", lw=1))
    ax.plot([x + 0.35, x + 0.52], [y + 0.75, y + 0.9], color=VERMILLION, lw=1.5)
    ax.add_patch(FancyBboxPatch((x + 0.85, y + 0.25), 0.9, 1.0, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="#D9D9D9", ec="#404040", lw=1))
    ax.plot([x + 0.35, x + 0.35, x + 0.85], [y + 0.42, y + 0.2, y + 0.45], color="#404040", lw=1)


def cm_to_fig(x, y, w, h):
    return [x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT]


def arrow(x0, x1, y=3.9):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=14, color="black", lw=1.5))


# Block 1: population and data
x1 = 0.75
ax.text(x1, 6.6, "Population and data", fontsize=11, fontweight="bold")
ax.text(x1, 6.2, "79,858 adults, emergency\nand intensive care", fontsize=10, va="top", color="#404040", linespacing=1.2)
cuff(x1, 4.05)
ax.text(x1 + 1.95, 4.55, "Blood\npressure", fontsize=10, va="center")
tube(x1 + 0.2, 2.55)
tube(x1 + 0.7, 2.55)
ax.text(x1 + 1.95, 3.15, "17 blood\ntests", fontsize=10, va="center")
ax.text(x1, 2.1, "Metabolic panel\nand blood count,\n6 statistics each", fontsize=10, va="top", linespacing=1.2)

# Block 2: comparison
x2, w2 = 5.35, 3.7
arrow(4.55, x2 - 0.15)
ax.text(x2, 6.6, "Comparison", fontsize=11, fontweight="bold")
ax.add_patch(FancyBboxPatch((x2, 4.6), w2, 1.1, boxstyle="round,pad=0,rounding_size=0.15", fc="#F2F2F2", ec=GREY, lw=1))
ax.text(x2 + w2 / 2, 5.15, "Blood pressure alone", fontsize=10, ha="center", va="center")
ax.text(x2 + w2 / 2, 4.2, "vs", fontsize=10, ha="center", va="center", fontstyle="italic")
ax.add_patch(FancyBboxPatch((x2, 2.2), w2, 1.6, boxstyle="round,pad=0,rounding_size=0.15", fc="#E5F1F8", ec=BLUE, lw=1))
ax.text(x2 + w2 / 2, 3.0, "Blood pressure +\nlaboratory predictors\n(30, 70 or all 102)", fontsize=10,
        ha="center", va="center", linespacing=1.2)
ax.text(x2, 1.55, "5 algorithms, tested on\n19,965 unseen patients", fontsize=10, va="top", color="#404040", linespacing=1.2)

# Block 3: findings (discrimination and main analytes)
x3 = 9.75
arrow(x2 + w2 + 0.1, x3 - 0.1)
ax.text(x3, 6.6, "Findings", fontsize=11, fontweight="bold")

ax.text(x3, 6.05, "Area under the ROC curve", fontsize=10, va="center")
axA = fig.add_axes(cm_to_fig(12.9, 4.25, 3.9, 1.5))
names, values, colors = zip(*AUROC)
axA.barh(range(4)[::-1], values, color=colors, height=0.65)
for yy, v in zip(range(4)[::-1], values):
    axA.text(v + 0.002, yy, f"{v:.3f}", va="center", fontsize=9)
axA.set_yticks(range(4)[::-1], names, fontsize=9)
axA.set_xlim(0.80, 0.90)
axA.set_xticks([0.80, 0.85, 0.90])

ax.text(x3, 3.3, "Largest contributions (%)", fontsize=10, va="center")
axB = fig.add_axes(cm_to_fig(12.9, 0.8, 3.9, 2.2))
n, p, g = zip(*ANALYTES_GA)
axB.barh(range(len(n))[::-1], p, color=[GROUP[x] for x in g], height=0.65)
axB.set_yticks(range(len(n))[::-1], n, fontsize=9)
axB.set_xlim(0, 18)
axB.set_xticks([0, 5, 10, 15])

for a in (axA, axB):
    a.tick_params(labelsize=8, length=2)
    a.spines[["top", "right"]].set_visible(False)

fig.savefig(SVG)
fig.savefig(TIFF, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print(f"Saved: {SVG} and {TIFF}")
