"""
Offline selective-prediction analysis over saved run logits.

Usage:
    python analyze_selective.py results/metaclip_fullcc/k16
    python analyze_selective.py results/laion2b_s32b_b79k/k16 --method combproto
    python analyze_selective.py RESULTS_DIR --max-runs 5      # quick look


--method selects which fused model to analyze and which two branches feed
the consensus/agreement analyses:
    tip_f     : tip_f_logits    = clip branch + fine-tuned cache branch
    tip       : tip_logits      = clip branch + training-free cache branch
    combproto : combproto_logits, branches = clip (text) + visual prototypes

Outputs (printed + saved to <dir>/analysis_<method>/):
  1. risk_coverage.png, two panels:
     (a) mean curves for all scores; std band ONLY for branch-consensus
         (+ zero-shot MSP reference)
     (b) paired per-run difference vs MSP (mean +- std) -- tight bands,
         shows whether small score differences are consistent
  2. Table A: selective accuracy at fixed coverage {50,70,90}%
  3. Table B: max coverage at target accuracy {80,85,90}%
  4. AURC and E-AURC (excess over per-run oracle; headroom-corrected,
     comparable across models); zero-shot reference included
  5. Paired comparison branch-consensus vs MSP (AURC/E-AURC diffs, wins)
  6. Binary branch agreement (accuracy when branches agree vs disagree)
  7. Branch prediction-distribution diagnostic (argmax share per class and
     branch vs true base rates)
  8. Per-class deferral RATES (fraction of each class deferred) and
     deferred-set composition, at DEFER_COVERAGE using --defer-score
  9. Aggregated row-normalized confusion matrix (print + confusion.png)

Notes:
  - Confidence scores are used as RANKINGS only. Fused logits carry an
    arbitrary temperature (100*cosine + cache terms), so absolute softmax
    values are not calibrated probabilities; do not interpret them as such.
  - Test sets differ slightly across runs (complement of the sampled
    support set); per-run computation + aggregation handles this.
  - AURC / E-AURC are computed EXACTLY, following Geifman et al. (2019):
    selective risk averaged over ALL n coverage levels (k/n, k = 1..n), with
    the oracle computed on the same n levels. Tied confidence values are
    handled by their expected risk under random tie-breaking. The 20-point
    COVERAGE_GRID is now used only for plotting and for the operating-point
    tables (A and B), whose values do not depend on how the area is integrated.
  - The zero-shot reference uses MSP, both
    in the figure and in the E-AURC table. 
  - Softmax is computed in float64 (float32 saturates MSP at exactly 1.0
    once the top logit leads by ~17; checked: no saturation on this data).
  - Reported standard deviations use the sample convention (ddof = 1)
    everywhere, set by STD_DDOF.
"""

import os
import glob
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------------------- publication styling (Elsevier line art) -----
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
MM = 1 / 25.4
RC_W_MM, RC_H_MM = 190, 78      # risk-coverage: double column
CM_W_MM, CM_H_MM = 100, 88      # confusion matrix: ~single column

# Okabe-Ito colourblind-safe palette + line styles per score
SCORE_STYLE = {
    "MSP":              dict(color="#0072B2", ls="-"),
    "margin":           dict(color="#E69F00", ls="--"),
    "neg-entropy":      dict(color="#009E73", ls=":"),
    "branch-consensus": dict(color="#D55E00", ls="-"),
}

# short display names for figure axes (fallback: raw class name)
CLASS_DISPLAY = {
    "Corrosion": "Corrosion",
    "Detachment": "Detachment",
    "Cracks": "Crack",
    "Stains": "Stain",
    "Spalling": "Spalling",
}

COVERAGE_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)   # plotting + Tables A/B only
REPORT_COVERAGES = [0.50, 0.70, 0.90]
TARGET_ACCS = [80.0, 85.0, 90.0]
DEFER_COVERAGE = 0.70
STD_DDOF = 1                    # sample standard deviation in all reported +- values

# method -> (fused key, branch A key/name, branch B key/name)
METHOD_CFG = {
    "tip_f": ("tip_f_logits", ("clip_logits", "text"), ("cache_logits_f", "cache")),
    "tip": ("tip_logits", ("clip_logits", "text"), ("cache_logits_tf", "cache")),
    "combproto": ("combproto_logits", ("clip_logits", "text"), ("proto_logits", "proto")),
}


# ----------------------------------------------------------------------
# Confidence scores (higher = more confident)
# ----------------------------------------------------------------------
def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)          # float64: avoids MSP saturating at 1.0
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def score_msp(logits, **kw):
    return softmax(logits).max(axis=1)


def score_margin(logits, **kw):
    s = np.sort(np.asarray(logits, dtype=np.float64), axis=1)
    return s[:, -1] - s[:, -2]                   # LOGIT margin (not probability margin)


def score_neg_entropy(logits, **kw):
    p = softmax(logits)
    return (p * np.log(p + 1e-12)).sum(axis=1)


def score_branch_consensus(logits, branch_a=None, branch_b=None, **kw):
    """Mean probability the two branches assign to the FUSED prediction."""
    pred = logits.argmax(axis=1)
    pa = softmax(branch_a)[np.arange(len(pred)), pred]
    pb = softmax(branch_b)[np.arange(len(pred)), pred]
    return 0.5 * (pa + pb)


SCORES = {
    "MSP": score_msp,
    "margin": score_margin,
    "neg-entropy": score_neg_entropy,
    "branch-consensus": score_branch_consensus,
}


# ----------------------------------------------------------------------
# Risk-coverage machinery
# ----------------------------------------------------------------------
def risk_coverage_curve(conf, correct, grid=COVERAGE_GRID):
    """Selective accuracy (%) at the grid coverages. Used for PLOTS and Tables A/B only."""
    order = np.argsort(-conf, kind="mergesort")
    cs = correct[order]
    n = len(conf)
    return np.array([cs[:max(1, int(round(c * n)))].mean() * 100 for c in grid])


def oracle_curve(base_acc_pct, grid=COVERAGE_GRID):
    """Selective accuracy of a perfect ranking (all errors last), for plotting."""
    a = base_acc_pct / 100.0
    return np.array([100.0 if c <= a else 100.0 * a / c for c in grid])


def aurc_exact(conf, correct):
    """Exact AURC and E-AURC (both in %, i.e. risk x100), Geifman et al. (2019).

    Sort by confidence (descending). At coverage k/n (k = 1..n) the selective
    risk is the error rate among the k most confident samples; AURC is the mean
    of these n risks. The oracle puts all correct predictions first, giving
    risk 0 up to coverage = accuracy and (k - n_correct)/k afterwards.
    E-AURC = AURC - oracle AURC. Tied confidences are assigned their expected
    risk under random tie-breaking, so the result does not depend on how
    argsort happens to order ties.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct).astype(bool)
    n = len(conf)
    order = np.argsort(-conf, kind="mergesort")
    s = conf[order]
    err = (~correct[order]).astype(np.float64)
    new_group = np.r_[True, s[1:] != s[:-1]]          # start of each block of tied scores
    gid = np.cumsum(new_group) - 1
    starts = np.flatnonzero(new_group)
    sizes = np.diff(np.r_[starts, n])
    g_err = np.add.reduceat(err, starts)              # errors inside each tie block
    before = np.r_[0.0, np.cumsum(g_err)[:-1]]        # errors in all earlier blocks
    pos = np.arange(n)
    exp_err = before[gid] + (pos - starts[gid] + 1) * g_err[gid] / sizes[gid]
    risk = exp_err / (pos + 1)
    k = np.arange(1, n + 1)
    oracle = np.maximum(0, k - correct.sum()) / k
    a = 100.0 * risk.mean()
    return a, a - 100.0 * oracle.mean()


def coverage_at_accuracy(conf, correct, target_acc):
    order = np.argsort(-conf, kind="mergesort")
    cs = correct[order]
    cum = np.cumsum(cs) / np.arange(1, len(cs) + 1) * 100
    ok = np.where(cum >= target_acc)[0]
    return (ok[-1] + 1) / len(cs) * 100 if len(ok) else 0.0


def sd(vals):
    v = np.asarray(vals, dtype=np.float64)
    return float(v.std(ddof=STD_DDOF)) if len(v) > STD_DDOF else 0.0


def ms(vals):
    v = np.array(vals)
    return f"{v.mean():5.1f}+-{sd(v):4.1f}"


def ms2(vals):
    v = np.array(vals)
    return f"{v.mean():6.2f}+-{sd(v):4.2f}"


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="results directory containing config.json and runs/")
    ap.add_argument("--method", default="tip_f", choices=list(METHOD_CFG))
    ap.add_argument("--defer-score", default="branch-consensus", choices=list(SCORES))
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args()

    fused_key, (ka, name_a), (kb, name_b) = METHOD_CFG[args.method]
    analysis_dir = os.path.join(args.dir, f"analysis_{args.method}")
    os.makedirs(analysis_dir, exist_ok=True)

    with open(os.path.join(args.dir, "config.json")) as f:
        config = json.load(f)
    classes = config["classes"]
    C = len(classes)

    run_files = sorted(glob.glob(os.path.join(args.dir, "runs", "run_*.npz")))
    if args.max_runs:
        run_files = run_files[:args.max_runs]
    if not run_files:
        raise SystemExit(f"No run files found in {args.dir}/runs/")
    print(f"Analyzing {len(run_files)} runs of {args.method} "
          f"({config['model']} {config['pretrained']}, K={config['k_shot']})\n")

    curves = {n: [] for n in SCORES}
    curves_zs = []
    aurcs = {n: [] for n in SCORES}
    eaurcs = {n: [] for n in SCORES}
    aurc_zs, eaurc_zs, eaurc_zs_margin = [], [], []
    acc_at_cov = {n: {c: [] for c in REPORT_COVERAGES} for n in SCORES}
    cov_at_acc = {n: {t: [] for t in TARGET_ACCS} for n in SCORES}
    agree_stats = []
    branch_pred_share = {name_a: np.zeros(C), name_b: np.zeros(C)}
    n_pred_total = 0
    deferred_class = np.zeros(C)
    test_class = np.zeros(C)
    deferred_total = 0
    kept_class = np.zeros(C)
    kept_correct = np.zeros(C)
    deferred_correct = np.zeros(C)
    confmat = np.zeros((C, C))

    for rf in run_files:
        d = np.load(rf)
        if fused_key not in d:
            raise SystemExit(f"{rf} has no '{fused_key}' "
                             f"(available: {list(d.keys())})")
        logits, ba, bb = d[fused_key], d[ka], d[kb]
        labels = d["test_labels"]
        preds = logits.argmax(axis=1)
        correct = (preds == labels).astype(float)

        for n_, fn in SCORES.items():
            conf = fn(logits, branch_a=ba, branch_b=bb)
            rc = risk_coverage_curve(conf, correct)
            curves[n_].append(rc)
            a, ea = aurc_exact(conf, correct)
            aurcs[n_].append(a)
            eaurcs[n_].append(ea)
            for c in REPORT_COVERAGES:
                acc_at_cov[n_][c].append(rc[np.argmin(np.abs(COVERAGE_GRID - c))])
            for t in TARGET_ACCS:
                cov_at_acc[n_][t].append(coverage_at_accuracy(conf, correct, t))

        # zero-shot reference: MSP of the zero-shot (text) classifier + its E-AURC
        zs_logits = d["clip_logits"]
        zs_correct = (zs_logits.argmax(axis=1) == labels).astype(float)
        zs_conf = score_msp(zs_logits)
        curves_zs.append(risk_coverage_curve(zs_conf, zs_correct))
        a, ea = aurc_exact(zs_conf, zs_correct)
        aurc_zs.append(a)
        eaurc_zs.append(ea)
        eaurc_zs_margin.append(aurc_exact(score_margin(zs_logits), zs_correct)[1])

        # binary agreement between branches
        agree = ba.argmax(axis=1) == bb.argmax(axis=1)
        if agree.any() and (~agree).any():
            agree_stats.append((agree.mean() * 100,
                                correct[agree].mean() * 100,
                                correct[~agree].mean() * 100))

        # branch prediction-distribution diagnostic
        for key, arr in ((name_a, ba), (name_b, bb)):
            pr = arr.argmax(axis=1)
            for c in range(C):
                branch_pred_share[key][c] += (pr == c).sum()
        n_pred_total += len(labels)

        # deferral at DEFER_COVERAGE using --defer-score
        conf = SCORES[args.defer_score](logits, branch_a=ba, branch_b=bb)
        n_keep = int(round(DEFER_COVERAGE * len(conf)))
        order = np.argsort(-conf, kind="mergesort")
        kept, deferred = order[:n_keep], order[n_keep:]
        for c in range(C):
            mk, md = labels[kept] == c, labels[deferred] == c
            kept_class[c] += mk.sum()
            kept_correct[c] += correct[kept][mk].sum()
            deferred_class[c] += md.sum()
            deferred_correct[c] += correct[deferred][md].sum()
            test_class[c] += (labels == c).sum()
        deferred_total += len(deferred)

        # confusion matrix (full coverage)
        for t, p in zip(labels, preds):
            confmat[t, p] += 1

    # ------------------------------------------------------------------
    # Figure: (a) curves, band only for consensus; (b) paired diff vs MSP
    # ------------------------------------------------------------------
    tag = f"{config['pretrained']}_{args.method}"
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(RC_W_MM * MM, RC_H_MM * MM))
    for n_ in SCORES:
        arr = np.stack(curves[n_])
        m = arr.mean(axis=0)
        st = SCORE_STYLE[n_]
        lw = 2.0 if n_ == "branch-consensus" else 1.6
        ax1.plot(COVERAGE_GRID * 100, m, lw=lw, label=n_,
                 color=st["color"], ls=st["ls"], zorder=3)
        if n_ == "branch-consensus":
            s = arr.std(axis=0, ddof=STD_DDOF)
            ax1.fill_between(COVERAGE_GRID * 100, m - s, m + s,
                             color=st["color"], alpha=0.15, lw=0, zorder=2)
    zs = np.stack(curves_zs)
    ax1.plot(COVERAGE_GRID * 100, zs.mean(axis=0), color="black",
             ls=(0, (4, 3)), lw=1.2, label="zero-shot (MSP)", zorder=1)
    ax1.set_xlabel("Coverage (%)")
    ax1.set_ylabel("Selective accuracy (%)")
    ax1.set_title("(a) Risk-coverage", pad=6)

    msp = np.stack(curves["MSP"])
    for n_ in SCORES:
        if n_ == "MSP":
            continue
        diff = np.stack(curves[n_]) - msp          # paired per run
        m, s = diff.mean(axis=0), diff.std(axis=0, ddof=STD_DDOF)
        st = SCORE_STYLE[n_]
        ax2.plot(COVERAGE_GRID * 100, m, lw=1.6, color=st["color"],
                 ls=st["ls"], zorder=3)
        ax2.fill_between(COVERAGE_GRID * 100, m - s, m + s,
                         color=st["color"], alpha=0.15, lw=0, zorder=2)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("Coverage (%)")
    ax2.set_ylabel(r"$\Delta$ selective accuracy vs MSP (pts)")
    ax2.set_title("(b) Paired difference vs MSP", pad=6)

    for ax in (ax1, ax2):
        ax.grid(True, color="0.85", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    handles, leg_labels = ax1.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    fig.legend(handles, leg_labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False,
               handlelength=2.4, columnspacing=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(analysis_dir, f"Fig_riskcov_{tag}.{ext}"),
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"Saved {analysis_dir}/Fig_riskcov_{tag}.pdf/.png")

    # confusion matrix figure (single-column)
    cm = confmat / confmat.sum(axis=1, keepdims=True)
    disp = [CLASS_DISPLAY.get(c, c) for c in classes]
    figc, axc = plt.subplots(figsize=(CM_W_MM * MM, CM_H_MM * MM))
    im = axc.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    axc.set_xticks(range(C), disp, rotation=45, ha="right")
    axc.set_yticks(range(C), disp)
    for i in range(C):
        for j in range(C):
            axc.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                     fontsize=8,
                     color="white" if cm[i, j] > 0.5 else "black")
    axc.set_xlabel("Predicted class")
    axc.set_ylabel("True class")
    cb = figc.colorbar(im, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    figc.tight_layout()
    for ext in ("pdf", "png"):
        figc.savefig(os.path.join(analysis_dir, f"Fig_confusion_{tag}.{ext}"),
                     dpi=600 if ext == "png" else None)
    plt.close(figc)
    print(f"Saved {analysis_dir}/Fig_confusion_{tag}.pdf/.png\n")

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    print("Table A -- selective accuracy (%) at fixed coverage")
    print("score            " + "".join(f"   cov {int(c*100)}%   " for c in REPORT_COVERAGES))
    for n_ in SCORES:
        print(f"{n_:16s}" + "".join(f"  {ms(acc_at_cov[n_][c])}" for c in REPORT_COVERAGES))

    print("\nTable B -- max coverage (%) at target selective accuracy")
    print("score            " + "".join(f"   acc>={int(t)}%   " for t in TARGET_ACCS))
    for n_ in SCORES:
        print(f"{n_:16s}" + "".join(f"  {ms(cov_at_acc[n_][t])}" for t in TARGET_ACCS))

    print("\nAURC / E-AURC, exact over all n coverage levels (x100; lower is better;"
          " E-AURC is headroom-corrected)")
    for n_ in SCORES:
        print(f"  {n_:16s} AURC {ms2(aurcs[n_])}   E-AURC {ms2(eaurcs[n_])}")
    print(f"  {'zero-shot/MSP':16s} AURC {ms2(aurc_zs)}   E-AURC {ms2(eaurc_zs)}")
    print(f"  (zero-shot with logit margin, as in earlier drafts: E-AURC {ms2(eaurc_zs_margin)})")

    # paired comparison: consensus vs MSP (dAURC == dE-AURC: same oracle within a run)
    d_aurc = np.array(aurcs["branch-consensus"]) - np.array(aurcs["MSP"])
    d_ea = np.array(eaurcs["branch-consensus"]) - np.array(eaurcs["MSP"])
    wins = int((d_aurc < 0).sum())
    print(f"\nPaired branch-consensus vs MSP over {len(d_aurc)} runs "
          f"(negative = consensus better):")
    print(f"  dAURC   {d_aurc.mean():+5.2f} +- {sd(d_aurc):4.2f}   "
          f"consensus wins {wins}/{len(d_aurc)}")
    print(f"  dE-AURC {d_ea.mean():+5.2f} +- {sd(d_ea):4.2f}   "
          f"(identical to dAURC: both scores share the same predictions and oracle)")
    wilcoxon_p = None
    try:
        from scipy import stats
        w_stat, wilcoxon_p = stats.wilcoxon(d_aurc, alternative="two-sided")
        t_stat, t_p = stats.ttest_rel(d_aurc, np.zeros_like(d_aurc))
        print(f"  Wilcoxon signed-rank p = {wilcoxon_p:.2g}   "
              f"(paired t-test p = {t_p:.2g})")
    except ImportError:
        print("  (scipy not available: install it for the Wilcoxon test)")

    if agree_stats:
        a = np.array(agree_stats)
        print(f"\nBinary branch agreement ({name_a} vs {name_b} argmax):")
        print(f"  agree on {a[:, 0].mean():5.1f}% of samples -> acc {a[:, 1].mean():5.1f}%")
        print(f"  disagree on {100 - a[:, 0].mean():5.1f}% -> acc {a[:, 2].mean():5.1f}%")

    print("\nBranch prediction distribution (share of argmax per class, %)"
          "  [diagnostic: is a branch collapsed onto few classes?]")
    base = test_class / test_class.sum() * 100
    print(f"  {'class':32s} {'true base':>10s} {name_a:>10s} {name_b:>10s}")
    for c in range(C):
        sa = branch_pred_share[name_a][c] / n_pred_total * 100
        sb = branch_pred_share[name_b][c] / n_pred_total * 100
        print(f"  {classes[c]:32s} {base[c]:10.1f} {sa:10.1f} {sb:10.1f}")

    print(f"\nDeferral at {int(DEFER_COVERAGE*100)}% coverage "
          f"(score: {args.defer_score}):")
    print(f"  {'class':32s} {'defer rate':>11s} {'kept acc':>9s} "
          f"{'defer acc':>10s} {'of deferred':>12s} {'base rate':>10s}")
    for c in range(C):
        rate = deferred_class[c] / max(test_class[c], 1) * 100
        ka_ = kept_correct[c] / max(kept_class[c], 1) * 100
        da = deferred_correct[c] / max(deferred_class[c], 1) * 100
        comp = deferred_class[c] / max(deferred_total, 1) * 100
        print(f"  {classes[c]:32s} {rate:10.1f}% {ka_:8.1f}% {da:9.1f}% "
              f"{comp:11.1f}% {base[c]:9.1f}%")

    out = {
        "method": args.method, "n_runs": len(run_files),
        "aurc_definition": "exact: mean selective risk over all n coverage levels "
                           "(Geifman et al. 2019); ties at expected risk; x100",
        "std_ddof": STD_DDOF,
        "coverage_grid": COVERAGE_GRID.tolist(),
        "curves_mean": {n_: np.stack(curves[n_]).mean(axis=0).tolist() for n_ in SCORES},
        "curves_std": {n_: np.stack(curves[n_]).std(axis=0, ddof=STD_DDOF).tolist() for n_ in SCORES},
        "zeroshot_msp_curve_mean": np.stack(curves_zs).mean(axis=0).tolist(),
        "aurc": {n_: float(np.mean(aurcs[n_])) for n_ in SCORES},
        "aurc_std": {n_: sd(aurcs[n_]) for n_ in SCORES},
        "eaurc": {n_: float(np.mean(eaurcs[n_])) for n_ in SCORES},
        "eaurc_std": {n_: sd(eaurcs[n_]) for n_ in SCORES},
        "eaurc_zeroshot_msp": float(np.mean(eaurc_zs)),
        "eaurc_zeroshot_msp_std": sd(eaurc_zs),
        "eaurc_zeroshot_margin": float(np.mean(eaurc_zs_margin)),
        "paired_consensus_vs_msp": {"d_aurc_mean": float(d_aurc.mean()),
                                    "d_aurc_std": sd(d_aurc),
                                    "wins": wins, "n": len(d_aurc),
                                    "wilcoxon_p": (float(wilcoxon_p)
                                                   if wilcoxon_p is not None
                                                   else None)},
        "acc_at_coverage": {n_: {str(c): float(np.mean(v)) for c, v in d_.items()}
                            for n_, d_ in acc_at_cov.items()},
        "coverage_at_accuracy": {n_: {str(t): float(np.mean(v)) for t, v in d_.items()}
                                 for n_, d_ in cov_at_acc.items()},
        "coverage_at_accuracy_std": {n_: {str(t): sd(v) for t, v in d_.items()}
                                     for n_, d_ in cov_at_acc.items()},
        "deferral_rate_per_class": {classes[c]: float(deferred_class[c] / max(test_class[c], 1))
                                    for c in range(C)},
        "kept_acc_per_class": {classes[c]: float(kept_correct[c] / max(kept_class[c], 1))
                               for c in range(C)},
        "deferred_acc_per_class": {classes[c]: float(deferred_correct[c] / max(deferred_class[c], 1))
                                   for c in range(C)},
        "confusion_row_normalized": cm.tolist(),
    }
    with open(os.path.join(analysis_dir, "selective_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRaw summary saved to {analysis_dir}/selective_summary.json")


if __name__ == "__main__":
    main()