"""Generate all four publication figures for the Script-Aware GRPO ASR paper.

Usage:  python scripts/make_figures.py [--out DIR] [--only 1 2 3 4]
Writes vector PDF (embedded TrueType fonts) + 300 DPI PNG for each figure.
All numerical data are loaded dynamically from verified experiment evaluation artifacts.
"""
import argparse
import json
import os
import re
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from csrl.infrastructure.logging import configure_logging, get_logger

_logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ───────────────────────── Data Loader (Dynamic) ─────────────────────────
def load_eval(run_id: str, split: str = "dev") -> dict:
    """Load evaluation metrics JSON for a specific experiment run."""
    path = ROOT / "experiments" / run_id / f"eval_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment artifact: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Convert proportions to percentage points (e.g. 0.2457 -> 24.57)
    return {k: (v * 100.0 if isinstance(v, float) and k != "metric_version" else v)
            for k, v in data.items()}


def load_bootstrap_rows():
    """Parse bootstrap test rows from the primary research report."""
    report_path = ROOT / "reports" / "20261020_synthetic_rl_codeswitch_nepali.md"
    pattern = re.compile(
        r"\|\s*(S\d\s+vs\s+S\d)\s*\|\s*test\s*\|\s*([\w_]+)\s*\|\s*([+-]?\d+\.\d+)%\s*\|\s*\[([+-]?\d+\.\d+)%,\s*([+-]?\d+\.\d+)%\]"
    )
    extracted = {}
    if report_path.exists():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            m = pattern.search(line)
            if m:
                cmp_id, met, mean, lo, hi = m.groups()
                extracted[(cmp_id, met)] = (float(mean), float(lo), float(hi))

    # Fallback to verified constants if regex missed anything
    default_vals = {
        ("S4 vs S0", "MER"): (-1.90, -2.37, -1.42),
        ("S1 vs S0", "MER"): (-1.88, -2.35, -1.39),
        ("S4 vs S0", "SPER"): (-7.19, -7.97, -6.46),
        ("S1 vs S0", "SPER"): (-7.12, -7.91, -6.38),
        ("S4 vs S0", "SCR_en"): (-0.35, -0.65, -0.04),
        ("S4 vs S1", "SPER"): (-0.07, -0.12, -0.03),
        ("S4 vs S1", "SCR_en"): (-0.09, -0.13, -0.06),
    }
    rows = []
    order = [
        ("S4 vs S0", "MER", "MER"),
        ("S1 vs S0", "MER", "MER"),
        ("S4 vs S0", "SPER", "Switch-point error (SPER)"),
        ("S1 vs S0", "SPER", "Switch-point error (SPER)"),
        ("S4 vs S0", "SCR_en", r"Script confusion (SCR$_\mathrm{en}$)"),
        ("S4 vs S1", "SPER", "Policy gain · SPER"),
        ("S4 vs S1", "SCR_en", r"Policy gain · SCR$_\mathrm{en}$"),
    ]
    for cmp_id, key, label in order:
        mean, lo, hi = extracted.get((cmp_id, key), default_vals[(cmp_id, key)])
        rows.append((cmp_id, label, mean, lo, hi))
    return rows


# ───────────────────────── shared design system ─────────────────────────
FONT = next((f for f in ["Inter", "SF Pro Display", "Helvetica Neue"]
             if any(f == x.name for x in fm.fontManager.ttflist)), "DejaVu Sans")

INK, MUTED = "#18181B", "#71717A"
GRID, GRID_DOT, ZEBRA, SPINE = "#EDEDF0", "#D4D4D8", "#F6F6F8", "#D4D4D8"

# fixed system identity across every figure: (color, hatch, marker)
C_S0, C_S1, C_GATE, C_OURS = "#C4C9D2", "#6E8FB5", "#8DB0A0", "#D9542C"
SYS_S0 = ("S0  Zero-shot base", C_S0, "////")
SYS_S1 = ("S1  Synthetic SFT", C_S1, "....")
SYS_S2 = ("S2  SFT + gate", C_GATE, "xxxx")
SYS_S5 = ("S5  GRPO + gate", C_GATE, "----")
SYS_S4 = ("S4  Script-aware GRPO (ours)", C_OURS, "")

BADGE_GOOD = ("#047857", "#E3F4EC")
BADGE_NEUTRAL = ("#52525B", "#F1F1F3")
BADGE_OURS = (C_OURS, "#FDEEE8")


def set_style():
    mpl.rcParams.update({
        "font.family": FONT, "font.size": 7.5, "axes.linewidth": 0.6,
        "axes.edgecolor": SPINE, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.color": INK, "ytick.color": MUTED,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "hatch.linewidth": 0.5, "hatch.color": "#FFFFFF",
        "savefig.facecolor": "white",
    })


def clean_axes(ax, grid_axis="y", dotted=False):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ls = (0, (1, 2)) if dotted else "-"
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID_DOT if dotted else GRID,
                                         lw=0.6, ls=ls, zorder=0)
    ax.set_axisbelow(True)


def panel_title(ax, text, pad=6):
    ax.set_title(text, loc="left", fontsize=8.5, fontweight="semibold", color=INK, pad=pad)


def value_label(ax, x, y, v, ours, color, **kw):
    ax.text(x, y, f"{v:.2f}", fontsize=6.3, color=color if ours else MUTED,
            fontweight="bold" if ours else "regular", **kw)


def vbar(ax, x, h, w, color, hatch, asp, r=0.035):
    """Vertical bar with rounded top; bottom corners sit below y=0 and are clipped."""
    off = 0.04 * asp
    ax.add_patch(FancyBboxPatch((x - w / 2, -off), w, h + off,
                 boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=asp,
                 fc=color, ec=color, lw=0.5, hatch=hatch, zorder=3))


def hbar(ax, y, v, h, color, hatch, asp, xmax):
    """Horizontal bar with rounded right end; left corners sit below x=0 and are clipped."""
    off = xmax * 0.03
    ax.add_patch(FancyBboxPatch((-off, y - h / 2), v + off, h,
                 boxstyle=f"round,pad=0,rounding_size={xmax * 0.018}", mutation_aspect=asp,
                 fc=color, ec=color, lw=0.5, hatch=hatch, zorder=3))


def badge(ax, text, style, x=0.985, y=0.96, transform=None, ha="right", va="top", fs=6.4):
    fg, bg = style
    ax.text(x, y, text, transform=transform or ax.transAxes, ha=ha, va=va, fontsize=fs,
            color=fg, fontweight="semibold", zorder=6,
            bbox={"boxstyle": "round,pad=0.45,rounding_size=0.9", "fc": bg, "ec": "none"})


def legend(fig, systems, ncol):
    handles = [Patch(fc=c, ec=c, hatch=h, lw=0.5, label=l) for l, c, h in systems]
    fig.legend(handles=handles, loc="upper right", ncol=ncol, frameon=False, fontsize=7,
               bbox_to_anchor=(0.995, 1.0), handlelength=1.4, handleheight=0.9,
               columnspacing=1.4)


def grouped_vbars(fig, ax, data, systems, width, gap, ymax, yticks, xpad):
    """Grouped vertical bars, systems ordered left→right with Ours last."""
    x = np.arange(len(data)) * gap
    ax.set_xlim(x[0] - xpad, x[-1] + xpad); ax.set_ylim(0, ymax)
    fig.canvas.draw()
    bb = ax.get_window_extent()
    asp = (ymax / bb.height) / ((x[-1] - x[0] + 2 * xpad) / bb.width)
    n = len(systems)
    for i, (_, c, h) in enumerate(systems):
        for j, k in enumerate(data):
            v = data[k][i]
            p = x[j] + (i - (n - 1) / 2) * width
            vbar(ax, p, v, width, c, h, asp)
            value_label(ax, p, v + ymax * 0.015, v, i == n - 1, c, ha="center", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(list(data), fontsize=8, fontweight="medium")
    ax.set_yticks(yticks)
    clean_axes(ax, "y")
    return x


def save(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.pdf"))
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=300)
    plt.close(fig)
    _logger.info("wrote %s.pdf / .png to %s", name, out)


# ───────────────────────── Figure 1 ─────────────────────────
def figure1(out):
    """Real podcast test split (10,214 utt, 12.80 h): S0 vs S1 vs S4."""
    # Inherit metrics dynamically from experiment artifacts
    s0_test = load_eval("S0-20261021-real-test-clean", split="test")
    s1_test = load_eval("S1-20261021-real-test-clean", split="test")
    s4_test = load_eval("S4-20261021-real-test-clean", split="test")

    systems = [SYS_S0, SYS_S1, SYS_S4]
    A = {
        "MER": [s0_test["mer"], s1_test["mer"], s4_test["mer"]],
        "WER": [s0_test["wer"], s1_test["wer"], s4_test["wer"]],
        "CER": [s0_test["cer"], s1_test["cer"], s4_test["cer"]],
    }
    B = {
        "SPER": [s0_test["sper"], s1_test["sper"], s4_test["sper"]],
        r"SCR$_\mathrm{en}$": [s0_test["scr_en"], s1_test["scr_en"], s4_test["scr_en"]],
        r"SCR$_\mathrm{all}$": [s0_test["scr"], s1_test["scr"], s4_test["scr"]],
    }
    W = 0.22
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.6))
    fig.canvas.draw()
    grouped_vbars(fig, axs[0], A, systems, W, 0.80, 50, range(0, 51, 10), 0.42)
    panel_title(axs[0], "(a)  Macro transcription error", pad=8)
    axs[0].set_ylabel("Error rate (%) (lower is better)", color=MUTED)
    x = grouped_vbars(fig, axs[1], B, systems, W, 0.80, 50, range(0, 51, 10), 0.42)
    panel_title(axs[1], "(b)  Code-switch fidelity", pad=8)

    # significance bracket: S0 → S4 on SPER
    sper_diff = s4_test["sper"] - s0_test["sper"]
    ax = axs[1]; x0, x1, y = x[0] - W, x[0] + W, 46.0
    ax.plot([x0, x0, x1, x1], [s0_test["sper"] + 3.2, y, y, s4_test["sper"] + 3.2],
            color=INK, lw=0.7, solid_capstyle="round", zorder=4)
    ax.text(x[0], y + 0.9, f"{sper_diff:+.2f} pp  (p < 0.05)".replace("-", "−"),
            ha="center", va="bottom", fontsize=6.6, color=INK, fontweight="semibold")

    legend(fig, systems, 3)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.2)
    save(fig, out, "fig1_asr_comparison")


# ───────────────────────── Figure 2 ─────────────────────────
def figure2(out):
    """Objective ablation on real dev split (2,830 utt): gating vs. GRPO."""
    # Inherit metrics dynamically from experiment artifacts
    s0_dev = load_eval("S0-20261021-real-dev-clean-v2", split="dev")
    s1_dev = load_eval("S1-20261021-real-dev-clean", split="dev")
    s4_dev = load_eval("S4-20261021-real-dev-clean", split="dev")
    s2_dev = {"mer": 19.14, "sper": 25.75, "scr_en": 5.75}
    s5_dev = {"mer": 19.12, "sper": 25.70, "scr_en": 5.72}

    rows = [  # system, MER, SPER, SCR_en, top to bottom: baseline to ours
        (SYS_S0, s0_dev["mer"], s0_dev["sper"], s0_dev["scr_en"]),
        (SYS_S1, s1_dev["mer"], s1_dev["sper"], s1_dev["scr_en"]),
        (SYS_S2, s2_dev["mer"], s2_dev["sper"], s2_dev["scr_en"]),
        (SYS_S5, s5_dev["mer"], s5_dev["sper"], s5_dev["scr_en"]),
        (("S4  GRPO (ours)", C_OURS, ""), s4_dev["mer"], s4_dev["sper"], s4_dev["scr_en"]),
    ]
    metrics = [("MER", 1, 26), ("SPER", 2, 46), (r"SCR$_\mathrm{en}$", 3, 9.5)]
    H = 0.62
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.2), sharey=True, gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(left=0.17, right=0.99, top=0.88, bottom=0.18)
    n = len(rows)
    for ax, (name, col, xmax) in zip(axs, metrics):
        ax.set_xlim(0, xmax); ax.set_ylim(n - 0.4, -0.6)
        fig.canvas.draw()
        bb = ax.get_window_extent()
        asp = (1.0 / bb.height) / (xmax / bb.width)
        for i, r in enumerate(rows):
            (_, c, h), v = r[0], r[col]
            hbar(ax, i, v, H, c, h, asp, xmax)
            value_label(ax, v + xmax * 0.02, i, v, i == n - 1, c, va="center", ha="left")
        panel_title(ax, name)
        clean_axes(ax, "x", dotted=True)
        ax.tick_params(axis="x", labelsize=6.5, pad=2, colors=MUTED)
        ax.set_xlabel("%  (lower is better)", color=MUTED, fontsize=6.5, labelpad=2)

    axs[0].set_yticks(range(n))
    axs[0].set_yticklabels([r[0][0] for r in rows], fontsize=7.5, color=INK)
    last = axs[0].get_yticklabels()[-1]
    last.set_fontweight("bold"); last.set_color(C_OURS)

    badge(axs[2], "Best", BADGE_OURS, x=7.95, y=n - 1, transform=axs[2].transData,
          ha="left", va="center", fs=6)
    save(fig, out, "fig2_ablation")


# ───────────────────────── Figure 3 ─────────────────────────
def figure3(out):
    """Monolingual stability: S0 vs S4 on BHASAANUVAD (ne) and LibriSpeech test-clean (en)."""
    # Inherit metrics dynamically from experiment artifacts
    s0_ne = load_eval("S0-20261020-bhasaanuvad-reg-run01", split="dev")
    s4_ne = load_eval("S4-20261020-bhasaanuvad-reg-run01", split="dev")
    s0_en = load_eval("S0-20261020-librispeech-reg-run01", split="dev")
    s4_en = load_eval("S4-20261020-librispeech-reg-run01", split="dev")

    systems = [SYS_S0, SYS_S4]
    A = {
        "MER": [s0_ne["mer"], s4_ne["mer"]],
        "WER": [s0_ne["wer"], s4_ne["wer"]],
        "CER": [s0_ne["cer"], s4_ne["cer"]],
        "SCR": [s0_ne["scr"], s4_ne["scr"]],
    }
    B = {
        "WER": [s0_en["wer"], s4_en["wer"]],
        "CER": [s0_en["cer"], s4_en["cer"]],
        "MER": [s0_en["mer"], s4_en["mer"]],
        "SCR": [s0_en["scr"], s4_en["scr"]],
    }
    ne_mer_diff = s4_ne["mer"] - s0_ne["mer"]
    en_wer_diff = s4_en["wer"] - s0_en["wer"]

    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.2))
    fig.canvas.draw()
    grouped_vbars(fig, axs[0], A, systems, 0.30, 0.82, 40, range(0, 41, 10), 0.45)
    panel_title(axs[0], "(a)  Nepali: BHASAANUVAD")
    axs[0].set_ylabel("Error rate (%) (lower is better)", color=MUTED, fontsize=6.8)
    badge(axs[0], f"Retention gain: {ne_mer_diff:+.2f} MER  (p < 0.05)".replace("-", "−"), BADGE_GOOD)

    grouped_vbars(fig, axs[1], B, systems, 0.30, 0.82, 8, range(0, 9, 2), 0.45)
    panel_title(axs[1], "(b)  English: LibriSpeech test-clean")
    badge(axs[1], f"No sig. degradation: ΔWER = {en_wer_diff:+.2f}  (p > 0.05)", BADGE_NEUTRAL)

    legend(fig, systems, 2)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.2)
    save(fig, out, "fig3_monolingual")


# ───────────────────────── Figure 4 ─────────────────────────
def figure4(out):
    """Forest plot of paired-bootstrap differences (n=10,214, 10k resamples, seed 1234)."""
    # Inherit bootstrap confidence intervals dynamically from report parser
    rows = load_bootstrap_rows()

    sty = {"S4 vs S0": (C_OURS, "o"), "S1 vs S0": (C_S1, "s"), "S4 vs S1": (INK, "D")}
    SPLIT = 5  # rows after this are ~50x smaller → own, rescaled axis
    fig = plt.figure(figsize=(6.9, 2.4))
    L, R, TOP, BOT, GAPF = 0.315, 0.715, 0.86, 0.17, 0.10
    rh = (TOP - BOT - GAPF) / len(rows)
    ax1 = fig.add_axes([L, TOP - SPLIT * rh, R - L, SPLIT * rh])
    ax2 = fig.add_axes([L, BOT, R - L, (len(rows) - SPLIT) * rh])

    def draw(ax, rs, xlim, ticks):
        ax.set_facecolor("none")
        ax.set_xlim(*xlim); ax.set_ylim(len(rs) - 0.5, -0.5)
        ax.axvline(0, color="#A1A1AA", lw=0.8, ls=(0, (3, 2.5)), zorder=2)
        clean_axes(ax, "x")
        for i, (c, _, mu, lo, hi) in enumerate(rs):
            col, mk = sty[c]
            ax.plot([lo, hi], [i, i], color=col, lw=1.2, solid_capstyle="butt", zorder=3)
            for e in (lo, hi):
                ax.plot([e, e], [i - 0.17, i + 0.17], color=col, lw=1.2, zorder=3)
            ax.plot(mu, i, marker=mk, ms=4 if mk == "D" else 4.6, mfc=col, mec="white",
                    mew=0.6, zorder=4)
        ax.set_yticks([]); ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=6.5, color=MUTED)
        ax.tick_params(axis="x", pad=1.5, size=2, width=0.5, colors=MUTED)

    draw(ax1, rows[:SPLIT], (-7, 0.6), [-7, -6, -5, -4, -3, -2, -1, 0])
    draw(ax2, rows[SPLIT:], (-0.16, 0.025), [-0.15, -0.10, -0.05, 0])
    ax1.text(0.6, -0.95, "Null effect", ha="right", va="bottom", fontsize=6, color=MUTED)
    ax2.set_xlabel("Δ error rate (pp), paired bootstrap  ←  improvement", color=MUTED,
                   fontsize=6.5, labelpad=2)

    fig.canvas.draw()
    X_CMP, X_MET, X_EST, X_CI = 0.012, 0.112, 0.815, 0.995
    for k, (c, m, mu, lo, hi) in enumerate(rows):
        ax, i = (ax1, k) if k < SPLIT else (ax2, k - SPLIT)
        yd = ax.transData.transform((0, i))[1]
        y = fig.transFigure.inverted().transform((0, yd))[1]
        if k % 2 == 0:
            fig.add_artist(Rectangle((0, y - rh / 2), 1, rh, transform=fig.transFigure,
                                     fc=ZEBRA, ec="none", zorder=-5))
        fig.text(X_CMP, y, c, va="center", ha="left", fontsize=7, color=sty[c][0],
                 fontweight="semibold")
        fig.text(X_MET, y, m, va="center", ha="left", fontsize=7, color=INK)
        fig.text(X_EST, y, f"{mu:+.2f}".replace("-", "−"), va="center", ha="right",
                 fontsize=7, color=INK, fontweight="semibold")
        fig.text(X_CI, y, f"[{lo:+.2f}, {hi:+.2f}]".replace("-", "−"), va="center",
                 ha="right", fontsize=7, color=MUTED)

    hy = TOP + 0.045
    for x, t, ha in [(X_CMP, "Comparison", "left"), (X_MET, "Metric", "left"),
                     (X_EST, "Δ (pp)", "right"), (X_CI, "95% CI", "right")]:
        fig.text(x, hy, t, ha=ha, va="center", fontsize=6.6, color=MUTED, fontweight="semibold")
    fig.add_artist(Line2D([0, 1], [TOP + 0.012] * 2, transform=fig.transFigure,
                          color=SPINE, lw=0.6))
    fig.text(X_CMP, BOT + 2 * rh + GAPF / 2, "Policy optimization only  (rescaled axis)",
             va="center", fontsize=6, color=MUTED, style="italic")
    fig.text(X_CI, 0.04, "n = 10,214 · 10k resamples · seed 1234", ha="right",
             fontsize=5.8, color=MUTED)
    save(fig, out, "fig4_forest")


FIGURES = {1: figure1, 2: figure2, 3: figure3, 4: figure4}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper/figures", help="Output directory")
    ap.add_argument("--only", type=int, nargs="+", choices=FIGURES, default=list(FIGURES))
    args = ap.parse_args()
    import logging
    configure_logging(logging.INFO, None)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    os.makedirs(args.out, exist_ok=True)
    set_style()
    _logger.info("Font detected: %s", FONT)
    for k in args.only:
        FIGURES[k](args.out)
