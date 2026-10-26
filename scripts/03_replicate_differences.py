"""What differs between the a/b libraries of the same stage and batch?

Gene-level evidence for the hypotheses of the report section on a/b
libraries (clutch effect, handling stress, developmental timing). Everything
is descriptive: with one library per side, there is no replicate-level
variance, so no p-values are reported.

  * Pseudobulk comparison: mean expression per 10k UMIs (each cell normalised
    to the same total first) and fraction of cells detecting each gene. The
    detection rates are computed after binomial thinning of the deeper
    library to the median depth of the other one (detection depends on depth).
  * On/off genes: detected in >= 40% of the cells of one library and <= 10%
    of the other (presence/absence, typical of genetic differences).
  * Two timing indicators, each expressed in units of one stage step:
      - ribosomal fraction: share of UMIs from ribosomal protein genes, which
        increases monotonically with developmental time in this atlas;
      - stage-marker score: log2 ratio of UMIs in "older" vs "younger" genes,
        defined from the pair and an adjacent stage of the same batch
        (both a and b must differ from the reference by > 2-fold in the same
        direction; ribosomal genes excluded). For high and oblong, the
        reference is dome, so this is a zygotic-genome-activation score.
    The shift (b - a) is divided by the difference between the older and the
    younger side (reference vs. pair), so +0.25 means "b is a quarter of a
    stage step older than a". An indicator is not used (n/a) if it has no
    trend between the pair and the reference (the step is smaller than the
    a/b difference), or if the marker score has fewer than MIN_MARKER_GENES
    genes. Timing is claimed ("a older"/"b older") only if both indicators are
    available, agree in sign, and both shifts are at least MIN_SHIFT; if they
    agree in sign but one is smaller, the label is "weak"; if they disagree,
    "conflicting".

Outputs:
  results/tables/replicate_de_genes.tsv       (|log2FC| >= 1 per pair)
  results/tables/library_ribosomal.tsv        (all libraries, incl. depth correlation)
  results/tables/replicate_timing.tsv
  report/tables/replicate_summary.tex
  report/tables/replicate_genes.tex
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_utils import REPORT_TABLES, TABLES, ensure_dirs, load_dropseq_counts

# (pair, library a, library b, reference library, reference is later)
PAIRS = [
    ("high DS5", "ZFHIGH_WT_DS5", "ZFHIGH_WT_DS5b", "ZFDOME_WT_DS5", True),
    ("oblong DS5", "ZFOBLONG_WT_DS5", "ZFOBLONG_WT_DS5b", "ZFDOME_WT_DS5", True),
    ("30% epiboly DS5", "ZF30_WT_DS5", "ZF30_WT_DS5b", "ZFS_WT_DS5", True),
    ("50% epiboly DS4", "ZF50_WT_DS4", "ZF50_WT_DS4b", "ZF60_WT_DS4", True),
    ("bud DS2", "ZFB_WT_DS2", "ZFB_WT_DS2b", "ZF90_WT_DS2", False),
    ("6-somite DS5", "ZF6S_WT_DS5", "ZF6S_WT_DS5b", "ZF3S_WT_DS5", False),
]
MIN_MEAN = 0.5  # UMIs per 10k, in at least one of the compared libraries
MIN_SHIFT = 0.1  # stage steps
MIN_MARKER_GENES = 30
SEED = 0
ON, OFF = 0.4, 0.1  # detection thresholds for on/off genes

# Curated genes shown in the report, with the reading they support.
EVIDENCE = [
    ("high DS5", ["GBGT1L4", "SI:DKEY-261J4.3"], "on/off"),
    ("high DS5", ["NANOS3", "DND1"], "germ plasm"),
    ("high DS5", ["TBX16"], "early zygotic"),
    ("oblong DS5", ["ALDH1L1", "CHIA.6"], "near-absent in b"),
    ("oblong DS5", ["KRT8", "KRT18", "CLDNE"], "EVL"),
    ("50% epiboly DS4", ["HSP90AA1.2", "HSP70.3", "SERPINH1B"], "heat shock"),
    ("50% epiboly DS4", ["SI:DKEY-108K21.10"], "on/off"),
    ("bud DS2", ["MYOD1", "MYL10", "ACTC1A"], "muscle (adaxial)"),
    ("bud DS2", ["SOX19B"], "early neuroectoderm"),
]


def ribosomal_mask(genes: pd.Index) -> np.ndarray:
    upper = genes.str.upper()
    return np.asarray(upper.str.match(r"^RP[SL]\d+[A-Z]?\d*$") | upper.str.match(r"^RPLP\d"))


class Library:
    """Normalised expression of one library (cells x genes, per 10k UMIs)."""

    def __init__(self, adata, name: str):
        self.counts = adata.X[(adata.obs["library"] == name).to_numpy()].tocsr()
        totals = np.asarray(self.counts.sum(axis=1)).ravel()
        self.norm = self.counts.multiply(1e4 / totals[:, None]).tocsr()
        self.mean = np.asarray(self.norm.mean(axis=0)).ravel()
        self.n_cells = self.counts.shape[0]
        self.median_umis = float(np.median(totals))

    def detected(self, keep: float, rng: np.random.Generator) -> np.ndarray:
        """Fraction of cells detecting each gene, after thinning to `keep`."""
        counts = self.counts.copy()
        if keep < 1:
            counts.data = rng.binomial(counts.data.astype(np.int64), keep).astype(np.float32)
        return np.asarray((counts > 0).mean(axis=0)).ravel()

    def score(self, genes: np.ndarray) -> np.ndarray:
        """Per-cell UMIs (per 10k) in a gene set."""
        return np.asarray(self.norm[:, genes].sum(axis=1)).ravel()


def depth_matched_detection(a: Library, b: Library, rng: np.random.Generator):
    """Detection rates of a and b with the deeper library thinned."""
    low = min(a.median_umis, b.median_umis)
    return (a.detected(low / a.median_umis, rng), b.detected(low / b.median_umis, rng))


def lfc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """log2 fold change y / x of mean expression per 10k."""
    return np.log2((y + 0.1) / (x + 0.1))


def de_table(pair: str, a: Library, b: Library, genes: pd.Index,
             det_a: np.ndarray, det_b: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "pair": pair, "gene": genes, "mean_a": a.mean, "mean_b": b.mean,
        "log2fc_b_vs_a": lfc(a.mean, b.mean), "detected_a": det_a,
        "detected_b": det_b,
    })
    df["on_off"] = ((df.detected_a >= ON) & (df.detected_b <= OFF)) | (
        (df.detected_b >= ON) & (df.detected_a <= OFF))
    expressed = np.maximum(a.mean, b.mean) >= MIN_MEAN
    return df[expressed & ((df["log2fc_b_vs_a"].abs() >= 1) | df["on_off"])]


def timing(a: Library, b: Library, ref: Library, ref_later: bool, ribo: np.ndarray) -> dict:
    """Shift of b relative to a, in stage steps, for both indicators."""
    expressed = np.maximum.reduce([a.mean, b.mean, ref.mean]) >= MIN_MEAN
    up_a, up_b = lfc(a.mean, ref.mean), lfc(b.mean, ref.mean)  # reference vs pair
    ref_up = expressed & ~ribo & (up_a > 1) & (up_b > 1)
    ref_down = expressed & ~ribo & (up_a < -1) & (up_b < -1)
    older, younger = (ref_up, ref_down) if ref_later else (ref_down, ref_up)

    def marker(lib):
        return np.median(np.log2((lib.score(older) + 1) / (lib.score(younger) + 1)))

    def ribo_pct(lib):
        return np.median(lib.score(ribo) / 100)

    out = {"older_genes": int(older.sum()), "younger_genes": int(younger.sum())}
    for name, f in (("marker", marker), ("ribo", ribo_pct)):
        va, vb, vr = f(a), f(b), f(ref)
        pair_mean = (va + vb) / 2
        step = (vr - pair_mean) if ref_later else (pair_mean - vr)
        shift = (vb - va) / step if step > abs(vb - va) else np.nan
        out.update({f"{name}_a": va, f"{name}_b": vb, f"{name}_ref": vr,
                    f"{name}_shift": shift})
    if out["older_genes"] + out["younger_genes"] < MIN_MARKER_GENES:
        out["marker_shift"] = np.nan
    m, r = out["marker_shift"], out["ribo_shift"]
    if np.isnan(m) or np.isnan(r):
        out["timing"] = "n/a"
    elif np.sign(m) != np.sign(r):
        out["timing"] = "conflicting"
    else:
        older = "b older" if m > 0 else "a older"
        out["timing"] = older if min(abs(m), abs(r)) >= MIN_SHIFT else f"weak, {older}"
    return out


def tex_gene(gene: str) -> str:
    return r"\emph{" + gene.lower().replace("_", r"\_") + "}"


def tex_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def tex_shift(x: float) -> str:
    return "n/a" if np.isnan(x) else f"${x:+.2f}$"


def write_summary_tex(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrl}", r"\toprule",
        r"Pair & Cells (a/b) & On/off & Ribo.\ \% (a/b) & Ribo.\ shift"
        r" & Marker shift & Timing \\", r"\midrule",
    ]
    for _, r in summary.iterrows():
        pair = r.pair.replace("%", r"\%")
        lines.append(
            f"{pair} & {tex_int(r.n_a)}/{tex_int(r.n_b)} & {r.on_off} & "
            f"{r.ribo_a:.1f}/{r.ribo_b:.1f} & {tex_shift(r.ribo_shift)} & "
            f"{tex_shift(r.marker_shift)} & {r.timing} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (REPORT_TABLES / "replicate_summary.tex").write_text("\n".join(lines) + "\n")


def write_genes_tex(rows: list[dict]) -> None:
    lines = [
        r"\begin{tabular}{lllrrrrr}", r"\toprule",
        r"Pair & Gene & Reading & \multicolumn{2}{c}{Mean/10k (a/b)} & $\log_2$FC"
        r" & \multicolumn{2}{c}{Detected (a/b)} \\", r"\midrule",
    ]
    previous = None
    for r in rows:
        if previous is not None and r["pair"] != previous:
            lines.append(r"\addlinespace[2pt]")
        pair = r["pair"].replace("%", r"\%") if r["pair"] != previous else ""
        previous = r["pair"]
        lines.append(
            f"{pair} & {tex_gene(r['gene'])} & {r['reading']} & {r['mean_a']:.1f} & "
            f"{r['mean_b']:.1f} & ${r['lfc']:+.1f}$ & {100 * r['det_a']:.0f}\\% & "
            f"{100 * r['det_b']:.0f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (REPORT_TABLES / "replicate_genes.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    adata = load_dropseq_counts()
    genes = adata.var_names
    ribo = ribosomal_mask(genes)
    print(f"ribosomal protein genes: {ribo.sum()}")
    libraries = {name: Library(adata, name) for name in adata.obs["library"].unique()}

    ribo_table = pd.DataFrame([
        {"library": name, "stage": adata.obs.loc[adata.obs["library"] == name, "stage"].iloc[0],
         "n_cells": lib.n_cells, "median_umis": lib.median_umis,
         "median_ribosomal_pct": np.median(lib.score(ribo) / 100),
         "spearman_with_log_umis": pd.Series(lib.score(ribo)).corr(
             pd.Series(np.log(np.asarray(lib.counts.sum(axis=1)).ravel())), method="spearman")}
        for name, lib in libraries.items()
    ])
    print("median within-library Spearman(ribosomal %, log UMIs): "
          f"{ribo_table['spearman_with_log_umis'].median():.2f}")
    ribo_table.to_csv(TABLES / "library_ribosomal.tsv", sep="\t", index=False,
                      float_format="%.4g")

    rng = np.random.default_rng(SEED)
    de, summary, evidence = [], [], []
    gene_index = {g: i for i, g in enumerate(genes)}
    for pair, lib_a, lib_b, lib_ref, ref_later in PAIRS:
        a, b, ref = libraries[lib_a], libraries[lib_b], libraries[lib_ref]
        det_a, det_b = depth_matched_detection(a, b, rng)
        table = de_table(pair, a, b, genes, det_a, det_b)
        de.append(table)
        t = timing(a, b, ref, ref_later, ribo)
        summary.append({"pair": pair, "library_a": lib_a, "library_b": lib_b,
                        "reference": lib_ref, "n_a": a.n_cells, "n_b": b.n_cells,
                        "genes_lfc_above_1": int((table["log2fc_b_vs_a"].abs() >= 1).sum()),
                        "on_off": int(table["on_off"].sum()), **t})
        for p, gene_list, reading in EVIDENCE:
            if p != pair:
                continue
            for g in gene_list:
                i = gene_index[g]
                evidence.append({"pair": pair, "gene": g, "reading": reading,
                                 "mean_a": a.mean[i], "mean_b": b.mean[i],
                                 "lfc": lfc(a.mean[i], b.mean[i]),
                                 "det_a": det_a[i], "det_b": det_b[i]})

    pd.concat(de).sort_values(["pair", "log2fc_b_vs_a"]).to_csv(
        TABLES / "replicate_de_genes.tsv", sep="\t", index=False, float_format="%.4g")
    summary = pd.DataFrame(summary)
    summary.to_csv(TABLES / "replicate_timing.tsv", sep="\t", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(summary.drop(columns=["library_a", "library_b"]).round(3).to_string())
    print(pd.DataFrame(evidence).round(2).to_string())
    for pair, table in zip([p[0] for p in PAIRS], de):
        on = table[table["on_off"]]
        print(f"{pair}: on/off genes: {', '.join(on.gene)}")
    write_summary_tex(summary)
    write_genes_tex(evidence)


if __name__ == "__main__":
    main()
