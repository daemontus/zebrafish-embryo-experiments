"""Schematic maps of where the programs arise: shield (6 hpf) and bud (10 hpf).

The embryo is unrolled around its animal-vegetal axis, a cylindrical
("Mercator-like") projection: x is the angle from the dorsal midline (0) to the
ventral side (+-180); y runs from the margin (shield) or the vegetal pole /
posterior end (bud, 0) to the animal pole / anterior end (1). At shield, the
cells are drawn where they are before they involute: mesendoderm at the
margin, ectoderm above it, the yolk syncytial layer below. At bud, the
ectoderm (outer) and the mesoderm and endoderm (inner) lie on top of each
other and are drawn as two maps.

Everything drawn comes from two hand-curated tables, one row per decision,
with its source (bib keys of report/references.bib; "annotation": the
location column of data/curated/curated_programs.tsv), a status (verified,
partly, unverified) and, where not verified, how to verify it:
  data/curated/fate_map_territories.tsv   territories (angle range on one side,
      mirrored; height range)
  data/curated/fate_map_signals.tsv       signal profiles along the dorsal-
      ventral (dv) or animal-vegetal / anterior-posterior (av) axis; their
      shapes are illustrative
Unverified territories are drawn with a dashed outline.

Outputs: report/figures/fate_map.pdf, report/tables/fate_map.tex (appendix:
territories, programs, sources, status)
"""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from pipeline_utils import (
    REPORT_TABLES,
    REPORT_WIDTH_IN,
    ROOT,
    ensure_dirs,
    report_style,
    save_report_figure,
)

CURATED = ROOT / "data" / "curated"
TERRITORIES = CURATED / "fate_map_territories.tsv"
SIGNALS = CURATED / "fate_map_signals.tsv"
GROUP_COLOURS = {"forebrain": "#5499C7", "mhb": "#76B7DA", "neural": "#85C1E9",
                 "border": "#B39DDB", "nonneural": "#F5CBA7", "axial": "#E74C3C",
                 "adaxial": "#F1948A", "paraxial": "#E59866", "lateral": "#C39BD3",
                 "heart": "#AF7AC5", "head": "#D7BDE2", "tail": "#A9DFBF", "endo": "#F7DC6F",
                 "ysl": "#D5D8DC", "dfc": "#2E4053", "": "#FFFFFF"}
SIGNAL_COLOURS = {"BMP": "#1F618D", "organizer antagonists": "#C0392B", "Wnt8": "#7D3C98",
                  "Nodal": "#B03A2E", "FGF": "#B9770E", "Hedgehog": "#C0392B",
                  "Wnt, FGF (posterior)": "#7D3C98", "RA": "#B9770E", "Fgf8 (MHB)": "#148F77"}
PANELS = [("shield", "blastoderm", "(a) shield (6 hpf): the blastoderm before involution"),
          ("bud", "outer", "(b) bud (10 hpf): ectoderm (outer layer)"),
          ("bud", "inner", "(c) bud (10 hpf): mesoderm and endoderm (inner layer)")]
TEXT_SIZE = 5.8
FONT_SIZES = (5.8, 5.2, 4.6)  # tried in turn until a label fits its box
FIT = 0.92  # a label may use this share of its box's width and height


def candidates(territory: str):
    """Label variants, longest first: the full name and its short form (before
    a colon or parenthesis), each wrapped at several widths."""
    brief = territory.split(":")[0].split(" (")[0]
    out = []
    for text in dict.fromkeys([territory, brief]):
        for width in (40, 24, 16, 11, 8):
            out.append(textwrap.fill(text, width, break_long_words=False))
    return list(dict.fromkeys(out))


def fit_labels(fig, labels: list) -> dict:
    """Place every box label at the largest variant and size that fits the box
    as drawn; boxes where nothing fits get a number (per panel, shared by the
    two mirrored boxes of a territory). Returns {ax: [(number, territory)]}."""
    renderer = fig.canvas.get_renderer()
    numbers = {}
    for ax, territory, a, b, y0, y1, colour in labels:
        (x0d, y0d), (x1d, y1d) = ax.transData.transform([(a, y0), (b, y1)])
        box_w, box_h = abs(x1d - x0d) * FIT, abs(y1d - y0d) * FIT
        placed = None
        for size in FONT_SIZES:
            for rotation in (0, 90):
                for text in candidates(territory):
                    if rotation == 90 and "\n" in text:
                        continue
                    t = ax.text((a + b) / 2, (y0 + y1) / 2, text, ha="center", va="center",
                                fontsize=size, rotation=rotation, zorder=6, color=colour)
                    ext = t.get_window_extent(renderer)
                    if ext.width <= box_w and ext.height <= box_h:
                        placed = t
                        break
                    t.remove()
                if placed:
                    break
            if placed:
                break
        if placed is None:
            key = numbers.setdefault(ax, {})
            n = key.setdefault(territory, len(key) + 1)
            ax.text((a + b) / 2, (y0 + y1) / 2, str(n), ha="center", va="center",
                    fontsize=FONT_SIZES[-1], fontweight="bold", zorder=6, color=colour)
    return {ax: sorted((n, terr) for terr, n in key.items()) for ax, key in numbers.items()}


def draw_territories(ax, t: pd.DataFrame, stage: str, labels: list) -> None:
    """Boxes of one map; their labels are collected in `labels` and placed by
    fit_labels once the figure layout is fixed."""
    for r in t.itertuples():
        if r.x0 == r.x1:  # a point: dorsal forerunners / Kupffer's vesicle
            x = 0 if stage == "shield" else 14  # beside the tailbud's label at bud
            ax.scatter([x], [r.y0], marker="^", s=30, color=GROUP_COLOURS[r.group], zorder=7)
            ax.annotate(r.territory, xy=(x, r.y0), xytext=(45, r.y0 - 0.17 if stage == "shield"
                                                           else r.y0 - 0.1),
                        fontsize=TEXT_SIZE, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=GROUP_COLOURS[r.group]))
            continue
        if not r.group:  # a boundary, not a territory
            continue
        # one side, mirrored; a range starting at the midline is one box across it
        sides = [(-r.x1, r.x1)] if r.x0 == 0 else [(r.x0, r.x1), (-r.x1, -r.x0)]
        overlay = r.group == "endo" and stage == "bud"  # scattered among the mesoderm
        for a, b in sides:
            ax.add_patch(FancyBboxPatch(
                (a, r.y0), b - a, r.y1 - r.y0, boxstyle="round,pad=0,rounding_size=3",
                mutation_aspect=1 / 180,
                facecolor="none" if overlay else GROUP_COLOURS[r.group],
                edgecolor="#B7950B" if overlay else ("#555555" if r.status == "unverified"
                                                     else "white"),
                hatch="...." if overlay else None,
                linewidth=0.6 if r.status == "unverified" or overlay else 1.0,
                linestyle="--" if r.status == "unverified" else "-",
                alpha=0.9, zorder=4 if overlay else 2))
            if overlay:  # explained in the caption
                continue
            if r.group == "ysl":
                ax.text(-95, (r.y0 + r.y1) / 2, r.territory, ha="center", va="center",
                        fontsize=TEXT_SIZE, zorder=6)
                continue
            labels.append((ax, r.territory, a, b, r.y0, r.y1,
                           "white" if r.group == "axial" else "black"))


def draw_signals(top, left, s: pd.DataFrame) -> None:
    x = np.linspace(-180, 180, 361)
    y = np.linspace(0, 1, 201)
    handles = []
    for r in s.itertuples():
        colour = SIGNAL_COLOURS.get(r.signal, "#555555")
        centre, width = float(r.centre), float(r.width)
        if r.axis == "dv":
            # BMP: (angle / 180) ** width when width is an exponent (< 10), else linear
            v = (np.abs(x) / 180) ** (width if width < 10 else 1) if r.signal == "BMP" \
                else np.exp(-((np.abs(x) - centre) / width) ** 2)
            top.plot(x, v, color=colour, lw=1.1)
            ls = "-"
        else:
            v = np.exp(-y / width) if centre == 0 else np.exp(-((y - centre) / width) ** 2)
            left.plot(v, y, color=colour, lw=1.1, ls="--")
            ls = "--"
        handles.append(plt.Line2D([], [], color=colour, lw=1.1, ls=ls,
                                  label=r.label + (" (left)" if r.axis != "dv" else "")))
    top.legend(handles=handles, frameon=False, fontsize=TEXT_SIZE - 0.3, loc="lower left",
               bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=1.8, columnspacing=1.0)


def figure(t: pd.DataFrame, s: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(REPORT_WIDTH_IN, REPORT_WIDTH_IN * 1.42))
    outer = fig.add_gridspec(3, 1, hspace=0.62, height_ratios=[1.2, 1, 1])
    labels, panels = [], []
    for k, (stage, layer, title) in enumerate(PANELS):
        g = outer[k].subgridspec(2, 2, width_ratios=[0.09, 1], height_ratios=[0.16, 1],
                                 wspace=0.02, hspace=0.04)
        ax = fig.add_subplot(g[1, 1])
        top = fig.add_subplot(g[0, 1], sharex=ax)
        left = fig.add_subplot(g[1, 0], sharey=ax)
        draw_territories(ax, t[(t["stage"] == stage) & (t["layer"] == layer)], stage, labels)
        panels.append(ax)
        draw_signals(top, left, s[s["stage"] == stage])
        ax.set_xlim(-180, 180)
        ax.set_ylim(-0.26 if stage == "shield" else -0.12, 1.0)
        ax.set_xticks([-180, -90, 0, 90, 180], ["ventral", "lateral", "dorsal", "lateral", "ventral"],
                      fontsize=6)
        ax.set_yticks([])
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        if stage == "shield":
            ax.axhline(0, color="black", lw=1.0, zorder=5)
            ax.text(183, 0.0, "margin", fontsize=TEXT_SIZE, va="center")
            ax.text(183, 0.9, "animal\npole", fontsize=TEXT_SIZE, va="center")
        else:
            ax.text(183, 0.04, "posterior\n(vegetal\npole)", fontsize=TEXT_SIZE, va="center")
            ax.text(183, 0.9, "anterior\n(animal\npole)", fontsize=TEXT_SIZE, va="center")
        top.set_title(title, fontsize=7.5, loc="left", pad=24)
        top.tick_params(labelbottom=False, length=0)
        top.set_yticks([])
        left.set_xticks([])
        left.tick_params(labelleft=False, length=0)
        for side in ("left", "right", "top", "bottom"):
            top.spines[side].set_visible(side == "bottom")
            left.spines[side].set_visible(side == "right")
        left.set_xlim(1.15, 0)
    keys = fit_labels(fig, labels)
    for ax in panels:  # key of the numbered boxes, below the panel's axis labels
        if ax in keys:
            ax.text(0.0, -0.17, "numbered: " + "   ".join(f"{n}: {terr}" for n, terr in keys[ax]),
                    fontsize=TEXT_SIZE - 0.4, ha="left", va="top", transform=ax.transAxes,
                    clip_on=False)
    save_report_figure(fig, "fate_map")
    plt.close(fig)


def latex(t: pd.DataFrame) -> None:
    def tex(x):
        return str(x).replace("_", " ").replace("%", "\\%").replace("&", "\\&")

    def cite(basis):
        parts = str(basis).split(";")
        keys = [b for b in parts if b and b != "annotation"]
        out = ["annotation"] if "annotation" in parts else []
        if keys:
            out.append("\\cite{" + ",".join(keys) + "}")
        return ", ".join(out)
    lines = ["% Generated by scripts/30_fate_map.py; do not edit.",
             "\\begin{longtable}{@{}l>{\\raggedright\\arraybackslash}p{4.4cm}"
             ">{\\raggedright\\arraybackslash}p{4.2cm}>{\\raggedright\\arraybackslash}p{3.2cm}l@{}}",
             "\\caption{Territories of the schematic maps of Figure~\\ref{fig:fate-map} "
             "(\\protect\\nolinkurl{data/curated/fate_map_territories.tsv}, drawn by "
             "\\protect\\nolinkurl{scripts/30_fate_map.py}): the curated programs they "
             "correspond to, the source (annotation: the location column of "
             "Table~\\ref{tab:curated-programs}) and whether their position is verified against "
             "the source, partly (the source supports the order but not the drawn extent) or "
             "unverified (placed from the annotation; the table file says how to check "
             "it).}\\label{tab:fate-map}\\\\",
             "\\toprule", "Map & Territory & Programs & Source & Status \\\\", "\\midrule",
             "\\endfirsthead", "\\toprule", "Map & Territory & Programs & Source & Status \\\\",
             "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot"]
    for i, r in enumerate(t[t["group"].astype(str) != ""].itertuples()):
        where = "shield" if r.stage == "shield" else f"bud, {r.layer}"
        lines.append(("\\rowcolor{gray!10}" if i % 2 else "") + " & ".join([
            where, tex(r.territory), tex(str(r.programs).replace(";", ", ")) if r.programs else "--",
            cite(r.basis), r.status]) + " \\\\")
    lines.append("\\end{longtable}")
    (REPORT_TABLES / "fate_map.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ensure_dirs()
    report_style()
    territories = pd.read_csv(TERRITORIES, sep="\t", keep_default_na=False)
    signals = pd.read_csv(SIGNALS, sep="\t", keep_default_na=False)
    figure(territories, signals)
    latex(territories)
