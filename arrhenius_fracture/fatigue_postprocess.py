"""Post-processing and visualization for Arrhenius fatigue fracture runs.

This module intentionally handles both outputs currently used in the package:

1. K-controlled front-local fatigue runs from ``fatigue_sharp_front.py``
   - fatigue_v1_history.csv
   - summary.json

2. Full-field sharp-front / FEM runs from ``sharp_front.py``
   - steps_####K.csv
   - fronts_####K.csv
   - crack_path*_####K.csv

The plotting functions are conservative: they never change results and they are
safe to run repeatedly.  They create per-run history/proxy plots and sweep-level
DeltaK-N / optional S-N summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Use a non-interactive backend before importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv_dict(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def _col(rows: List[dict], name: str, default=np.nan) -> np.ndarray:
    vals = []
    for r in rows:
        v = r.get(name, "")
        try:
            vals.append(float(v))
        except Exception:
            vals.append(default)
    return np.asarray(vals, dtype=float)


def _safe_loglog(ax, x, y, *args, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if np.any(m):
        ax.loglog(x[m], y[m], *args, **kwargs)


def _safe_semilogx(ax, x, y, *args, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if np.any(m):
        ax.semilogx(x[m], y[m], *args, **kwargs)


def plot_v1_history(run_dir: str | Path) -> Optional[Path]:
    """Plot the front-local V1 history in one multi-panel figure."""
    run_dir = Path(run_dir)
    hist_path = run_dir / "fatigue_v1_history.csv"
    rows = _read_csv_dict(hist_path)
    if not rows:
        return None

    N = _col(rows, "cycles_total")
    if not np.any(np.isfinite(N)):
        # Older rows may only have per-block cycles.  Accumulate them.
        N = np.cumsum(np.nan_to_num(_col(rows, "cycles"), nan=0.0))

    B = _col(rows, "B")
    Nem = _col(rows, "N_em")
    sb = _col(rows, "sigma_back_Pa") / 1e9
    dG = _col(rows, "dG_emb_eV")
    a_um = _col(rows, "a_adv_m") * 1e6
    Kmax = _col(rows, "Kmax_Pa_sqrt_m") / 1e6
    dK = _col(rows, "DeltaK_Pa_sqrt_m") / 1e6
    mu_e = _col(rows, "mu_emit")
    mu_p = _col(rows, "mu_peierls")
    mu_t = _col(rows, "mu_taylor")
    store = _col(rows, "store_per_cycle")

    fig, axes = plt.subplots(4, 2, figsize=(11.5, 12), constrained_layout=True)
    ax = axes.ravel()

    _safe_semilogx(ax[0], N, B, lw=2)
    ax[0].axhline(1.0, ls="--", lw=1)
    ax[0].set_ylabel("cleavage clock B")

    _safe_semilogx(ax[1], N, a_um, lw=2)
    ax[1].set_ylabel("crack advance (um)")

    _safe_semilogx(ax[2], N, Nem, lw=2)
    ax[2].set_ylabel("retained N_em")

    _safe_semilogx(ax[3], N, sb, lw=2)
    ax[3].set_ylabel("back stress (GPa)")

    _safe_semilogx(ax[4], N, dG, lw=2)
    ax[4].set_ylabel("embrittlement dG (eV)")

    _safe_semilogx(ax[5], N, Kmax, lw=2, label="Kmax")
    _safe_semilogx(ax[5], N, dK, lw=2, ls="--", label="DeltaK")
    ax[5].set_ylabel("K (MPa sqrt(m))")
    ax[5].legend(fontsize=8)

    _safe_loglog(ax[6], N, mu_e, lw=1.8, label="emit")
    _safe_loglog(ax[6], N, mu_p, lw=1.8, label="Peierls")
    _safe_loglog(ax[6], N, mu_t, lw=1.8, label="Taylor")
    ax[6].set_ylabel("cycle hazard")
    ax[6].legend(fontsize=8)

    _safe_loglog(ax[7], N, np.abs(store), lw=2)
    ax[7].set_ylabel("stored events/cycle")

    for a in ax:
        a.set_xlabel("cycles")
        a.grid(False)
        a.tick_params(direction="out")

    title = run_dir.name
    try:
        with (run_dir / "summary.json").open() as fp:
            s = json.load(fp)
        title = f"{run_dir.name}: T={s.get('T_K','?')} K, N={s.get('cycles_total',0):.3g}, n_adv={s.get('n_adv','?')}"
    except Exception:
        pass
    fig.suptitle(title, fontsize=12)
    out = run_dir / "fatigue_v1_history.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def plot_v1_process_zone_proxy(run_dir: str | Path) -> Optional[Path]:
    """Draw a front-local crack/process-zone proxy for V1.

    This is not a full field.  It is an explicit front-local visualization: crack
    line, final tip position, process-zone halo radius r_eff, and text diagnostics.
    """
    run_dir = Path(run_dir)
    hist_path = run_dir / "fatigue_v1_history.csv"
    rows = _read_csv_dict(hist_path)
    if not rows:
        return None
    last = rows[-1]
    def fval(k, d=0.0):
        try: return float(last.get(k, d))
        except Exception: return d
    a = fval("a_adv_m", 0.0)
    r = fval("r_eff_m", 1e-6)
    dG = fval("dG_emb_eV", 0.0)
    Nem = fval("N_em", 0.0)
    B = fval("B", 0.0)
    N = fval("cycles_total", np.nan)
    Kmax = fval("Kmax_Pa_sqrt_m", 0.0) / 1e6
    dK = fval("DeltaK_Pa_sqrt_m", 0.0) / 1e6

    # Use a simple local coordinate frame around an initial precrack tip.
    x_tip0 = 0.0
    x_tip = x_tip0 + a
    span = max(5*r, a + 5*r, 20e-6)
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    ax.plot([-span, x_tip], [0, 0], lw=3)
    circ = plt.Circle((x_tip, 0), r, fill=False, lw=2)
    ax.add_patch(circ)
    # Color proxy halo by embrittlement via a filled transparent disk.
    alpha = max(0.08, min(0.6, dG / 0.1 if dG > 0 else 0.08))
    halo = plt.Circle((x_tip, 0), r, alpha=alpha)
    ax.add_patch(halo)
    ax.plot([x_tip], [0], marker="o", ms=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.35*span, 0.35*span)
    ax.set_xlabel("local x around crack tip (m)")
    ax.set_ylabel("local y (m)")
    ax.set_title("V1 front-local process-zone proxy, not a full fatigue field")
    txt = (f"N={N:.3g} cycles\nKmax={Kmax:.3g}, DeltaK={dK:.3g} MPa sqrt(m)\n"
           f"a_adv={a*1e6:.3g} um, r_eff={r*1e6:.3g} um\n"
           f"N_em={Nem:.3g}, dG_emb={dG:.3g} eV, B={B:.3g}")
    ax.text(0.99, 0.98, txt, va="top", ha="right", transform=ax.transAxes,
            fontsize=9, bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9))
    out = run_dir / "tip_process_zone_proxy.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def _find_v1_run_dirs(root: Path) -> List[Path]:
    return sorted({p.parent for p in root.rglob("fatigue_v1_history.csv")})


def _summary_from_v1_run(run_dir: Path) -> Optional[dict]:
    sp = run_dir / "summary.json"
    hp = run_dir / "fatigue_v1_history.csv"
    if not sp.exists() or not hp.exists():
        return None
    try:
        with sp.open() as fp:
            s = json.load(fp)
    except Exception:
        s = {}
    rows = _read_csv_dict(hp)
    if not rows:
        return None
    Kmax = np.nanmax(_col(rows, "Kmax_Pa_sqrt_m")) / 1e6
    dK = np.nanmax(_col(rows, "DeltaK_Pa_sqrt_m")) / 1e6
    R = np.nanmedian(_col(rows, "R"))
    fHz = np.nanmedian(_col(rows, "frequency_Hz"))
    T = float(s.get("T_K", np.nan))
    n_adv = int(s.get("n_adv", 0) or 0)
    cycles_total = float(s.get("cycles_total", np.nan))
    # first advance cycle if present
    N = _col(rows, "cycles_total")
    a = _col(rows, "a_adv_m")
    fired = _col(rows, "n_fire")
    idx = np.where((fired > 0) | (a > 0))[0]
    cycles_first = float(N[idx[0]]) if idx.size and np.isfinite(N[idx[0]]) else math.nan
    cycles_fail = cycles_first if np.isfinite(cycles_first) else cycles_total
    out = {
        "run_dir": str(run_dir), "T_K": T, "Kmax_MPa_sqrt_m": Kmax,
        "DeltaK_MPa_sqrt_m": dK, "R": R, "frequency_Hz": fHz,
        "cycles_to_first_advance": cycles_first,
        "cycles_to_target_or_end": cycles_fail,
        "cycles_total": cycles_total, "n_adv": n_adv,
        "a_final_m": float(s.get("a_adv_m", math.nan)),
        "B_final": float(s.get("B", math.nan)),
        "N_em_final": float(s.get("N_em", math.nan)),
        "sigma_back_final_Pa": float(s.get("sigma_back_Pa", math.nan)),
        "dG_emb_final_eV": float(s.get("dG_emb_eV", math.nan)),
    }
    return out


def summarize_v1_sweep(root: str | Path, *, a0_m: Optional[float] = None,
                       Y: float = 1.0, stress_axis: bool = True) -> Optional[Path]:
    root = Path(root)
    run_dirs = _find_v1_run_dirs(root)
    rows = []
    for rd in run_dirs:
        r = _summary_from_v1_run(rd)
        if r is not None:
            if a0_m is not None and a0_m > 0 and np.isfinite(r["DeltaK_MPa_sqrt_m"]):
                # DeltaK [MPa sqrt(m)] = Y*DeltaSigma[MPa]*sqrt(pi*a)
                dSigma = r["DeltaK_MPa_sqrt_m"] / max(Y * math.sqrt(math.pi*a0_m), 1e-300)
                r["sigma_a_MPa"] = 0.5 * dSigma
            else:
                r["sigma_a_MPa"] = math.nan
            rows.append(r)
    if not rows:
        return None
    rows.sort(key=lambda r: (r.get("T_K", 0), r.get("Kmax_MPa_sqrt_m", 0), r.get("run_dir", "")))
    cols = list(rows[0].keys())
    outcsv = root / "fatigue_sweep_summary.csv"
    with outcsv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

    # DeltaK-N plot
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    Ts = sorted({r["T_K"] for r in rows if np.isfinite(r["T_K"])})
    for T in Ts:
        rr = [r for r in rows if r["T_K"] == T]
        x = [r["cycles_to_target_or_end"] for r in rr]
        y = [r["DeltaK_MPa_sqrt_m"] for r in rr]
        _safe_semilogx(ax, x, y, marker="o", lw=1.8, label=f"T={T:g} K")
        # mark no-advance/end points open-ish by overlaying x marker
        for r in rr:
            if int(r.get("n_adv", 0)) <= 0:
                ax.plot(r["cycles_to_target_or_end"], r["DeltaK_MPa_sqrt_m"], marker="x", ms=8, linestyle="none")
    ax.set_xlabel("cycles to first advance or run end")
    ax.set_ylabel("Delta K (MPa sqrt(m))")
    ax.set_title("Fatigue crack-growth threshold proxy")
    if Ts: ax.legend(fontsize=8)
    out1 = root / "deltaK_N_curve.png"
    fig.savefig(out1, dpi=250); plt.close(fig)

    if stress_axis and a0_m is not None and a0_m > 0:
        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        for T in Ts:
            rr = [r for r in rows if r["T_K"] == T]
            x = [r["cycles_to_target_or_end"] for r in rr]
            y = [r["sigma_a_MPa"] for r in rr]
            _safe_semilogx(ax, x, y, marker="o", lw=1.8, label=f"T={T:g} K")
        ax.set_xlabel("cycles to first advance or run end")
        ax.set_ylabel("nominal stress amplitude sigma_a (MPa)")
        ax.set_title(f"S-N conversion: K=Y sigma sqrt(pi a), a0={a0_m:g} m, Y={Y:g}")
        if Ts: ax.legend(fontsize=8)
        out2 = root / "SN_curve.png"
        fig.savefig(out2, dpi=250); plt.close(fig)
    return outcsv


def plot_sharp2d_history(run_dir: str | Path) -> List[Path]:
    """Plot existing full-field sharp-front/FEM CSV histories if present."""
    run_dir = Path(run_dir)
    outs = []
    for step_csv in sorted(run_dir.glob("steps_*K.csv")):
        rows = _read_csv_dict(step_csv)
        if not rows: continue
        tag = step_csv.stem.replace("steps_", "")
        step = _col(rows, "step")
        KJ = _col(rows, "KJ_Pa_sqrtm")/1e6
        B = _col(rows, "B")
        Nem = _col(rows, "N_em")
        a = _col(rows, "a_tip_m")*1e3
        sb = _col(rows, "sigma_back_Pa")/1e9
        lc = _col(rows, "lambda_c")
        le = _col(rows, "lambda_e")
        nf = _col(rows, "n_fire")
        fig, ax = plt.subplots(4, 2, figsize=(11.5, 12), constrained_layout=True)
        axs = ax.ravel()
        axs[0].plot(step, KJ, lw=2); axs[0].set_ylabel("KJ (MPa sqrt(m))")
        axs[1].plot(step, a, lw=2); axs[1].set_ylabel("a_tip (mm)")
        axs[2].plot(step, B, lw=2); axs[2].axhline(1, ls="--", lw=1); axs[2].set_ylabel("B")
        axs[3].plot(step, Nem, lw=2); axs[3].set_ylabel("N_em")
        axs[4].plot(step, sb, lw=2); axs[4].set_ylabel("sigma_back (GPa)")
        _safe_loglog(axs[5], np.maximum(step, 1), lc, lw=1.5, label="cleave")
        _safe_loglog(axs[5], np.maximum(step, 1), le, lw=1.5, label="emit")
        axs[5].set_ylabel("rates (1/s)"); axs[5].legend(fontsize=8)
        axs[6].plot(step, nf, lw=1.5); axs[6].set_ylabel("n_fire")
        axs[7].axis("off")
        for a0 in axs[:-1]:
            a0.set_xlabel("step/block"); a0.grid(False); a0.tick_params(direction="out")
        fig.suptitle(f"Full-field sharp-front/FEM diagnostics {tag}")
        out = run_dir / f"sharp2d_history_{tag}.png"
        fig.savefig(out, dpi=250); plt.close(fig)
        outs.append(out)
    return outs


def plot_crack_path_overlays(run_dir: str | Path) -> List[Path]:
    run_dir = Path(run_dir)
    outs = []
    # Group crack_path files by temperature token.
    temp_tokens = set()
    for p in run_dir.glob("crack_path*_*K.csv"):
        m = re.search(r"_(\d+)K\.csv$", p.name)
        if m: temp_tokens.add(m.group(1))
    for tok in sorted(temp_tokens):
        files = sorted(run_dir.glob(f"crack_path*_{tok}K.csv"))
        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        anyp = False
        for p in files:
            try:
                data = np.loadtxt(p, delimiter=",", skiprows=1)
                if data.ndim == 1: data = data[None, :]
                if data.shape[0] == 0: continue
                ax.plot(data[:,0]*1e3, data[:,1]*1e3, marker="o", ms=3, lw=1.8, label=p.stem)
                anyp = True
            except Exception:
                continue
        if not anyp:
            plt.close(fig); continue
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        ax.set_title(f"Crack path overlay {tok} K")
        ax.legend(fontsize=6, loc="best")
        out = run_dir / f"crack_path_overlay_{tok}K.png"
        fig.savefig(out, dpi=250); plt.close(fig)
        outs.append(out)
    return outs


def process_directory(root: str | Path, *, a0_m: Optional[float] = None, Y: float = 1.0) -> List[Path]:
    root = Path(root)
    made: List[Path] = []
    for rd in _find_v1_run_dirs(root):
        for fn in (plot_v1_history, plot_v1_process_zone_proxy):
            out = fn(rd)
            if out is not None: made.append(out)
    csvp = summarize_v1_sweep(root, a0_m=a0_m, Y=Y)
    if csvp is not None: made.append(csvp)
    made.extend(plot_sharp2d_history(root))
    made.extend(plot_crack_path_overlays(root))
    return made


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate fatigue fracture plots and sweep summaries.")
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--a0-m", type=float, default=None, help="Initial crack length for optional S-N conversion.")
    ap.add_argument("--Y", type=float, default=1.0, help="Geometry factor for optional S-N conversion.")
    args = ap.parse_args(argv)
    made = process_directory(args.root, a0_m=args.a0_m, Y=args.Y)
    if made:
        print("Generated:")
        for p in made:
            print(f"  {p}")
    else:
        print("No recognized fatigue output files found.")


if __name__ == "__main__":
    main()
