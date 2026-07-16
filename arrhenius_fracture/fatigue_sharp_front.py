"""Command-line smoke/demo driver for fatigue_v1.

Example:
  python -m arrhenius_fracture.fatigue_sharp_front \
    --temperatures 300 500 700 --Kmax-MPa-sqrt-m 20 --R 0.1 \
    --frequency-Hz 1000 --cycles-max 1e8 --out runs/fatigue_v1_demo
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np

from .config import ElasticProperties
from .sharp_front import (FrontConfig, FrontEngine, default_cleavage_barrier,
                          default_emission_barrier, apply_cleavage_barrier_args,
                          build_engine)
from .fatigue_v1 import (
    ExpFloorBarrierParams, ScaledExpFloorBarrier,
    FatigueWaveform, FatigueControllerConfig, FatigueCycleHazardController,
    build_controller_from_namespace, write_history_csv,
)


def _build_front(args, mat: ElasticProperties) -> FrontEngine:
    # v10 material classes use the same unified spatial MPZ state as monotonic
    # sharp-front fracture. Legacy scalar construction remains available only
    # when no material manifest/class is supplied.
    return build_engine(args, mat)


def _build_controller(args) -> FatigueCycleHazardController:
    return build_controller_from_namespace(args)


def run_one_temperature(args, T_K: float) -> dict:
    mat = ElasticProperties()
    front = _build_front(args, mat)
    controller = _build_controller(args)
    Kmax = args.Kmax_MPa_sqrt_m * 1.0e6
    wave = FatigueWaveform(Kmax=Kmax, R=args.R, frequency_Hz=args.frequency_Hz,
                           closure_clip=not args.no_closure_clip)

    outdir = os.path.join(args.out, f"T{int(round(T_K))}K")
    os.makedirs(outdir, exist_ok=True)
    controller.write_config(os.path.join(outdir, "fatigue_v1_controller_config.json"))

    args_payload = vars(args).copy()
    args_payload["T_K"] = T_K
    with open(os.path.join(outdir, "run_args.json"), "w") as fp:
        json.dump(args_payload, fp, indent=2, sort_keys=True)

    rows = []
    cycles_done = 0.0
    for ib in range(args.max_blocks):
        if cycles_done >= args.cycles_max:
            break
        remaining = max(float(args.cycles_max) - cycles_done, 0.0)
        req = min(args.block_cycles, remaining)
        # In hazard_limited mode the controller's max_block_cycles is the upper
        # bound.  Cap it by the remaining physical cycle horizon so a single
        # low-hazard VHCF block can jump exactly to cycles_max, but never beyond.
        old_max_block = float(controller.cfg.max_block_cycles)
        try:
            if np.isfinite(remaining):
                controller.cfg.max_block_cycles = min(old_max_block, remaining) if np.isfinite(old_max_block) else remaining
            row = controller.cycle_step_front(front, wave, T_K, requested_cycles=req)
        finally:
            controller.cfg.max_block_cycles = old_max_block
        row["block"] = ib
        cycles_done += row["cycles"]
        row["cycles_total"] = cycles_done
        rows.append(row)
        if args.print_every > 0 and (ib % args.print_every == 0 or row.get("fired", False)):
            print(
                f"T={T_K:g}K block={ib:05d} N={cycles_done:.3e} "
                f"dN={row['cycles']:.3g} B={row['B']:.3g} N_em={row['N_em']:.3g} "
                f"dG_emb={row['dG_emb_eV']:.3g}eV fired={row['n_fire']} a={row['a_adv_m']*1e6:.3f}um"
            )
        # For the first implementation, stop immediately on fire so the 2-D
        # integration can remesh/recompute K.  A 1-D demonstration can continue
        # if requested.
        if row.get("fired", False) and not args.continue_after_fire:
            break
        if front.n_adv >= args.n_advances:
            break

    write_history_csv(os.path.join(outdir, "fatigue_v1_history.csv"), rows)
    summary = {
        "T_K": T_K,
        "cycles_total": cycles_done,
        "n_blocks": len(rows),
        "n_adv": int(front.n_adv),
        "a_adv_m": float(front.a_adv),
        "B": float(front.B),
        "N_em": float(front.N_em),
        "r_eff_m": float(front.r_eff()),
        "sigma_back_Pa": float(front.sigma_back()),
        "dG_emb_eV": float(front.dG_emb() / 1.602176634e-19),
        "history_csv": os.path.join(outdir, "fatigue_v1_history.csv"),
    }
    with open(os.path.join(outdir, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    if not getattr(args, "no_plots", False):
        try:
            from .fatigue_postprocess import plot_v1_history, plot_v1_process_zone_proxy
            plot_v1_history(outdir)
            plot_v1_process_zone_proxy(outdir)
        except Exception as exc:
            print(f"  WARNING: fatigue plotting failed for {outdir}: {exc}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Version-1 fatigue process-zone hazard controller demo.")
    p.add_argument("--temperatures", type=float, nargs="+", default=[300.0])
    p.add_argument("--out", default="runs/fatigue_v1")
    p.add_argument("--material-class", choices=["ceramic", "weakT", "DBTT"], default=None)
    p.add_argument("--material-manifest", default=None)
    p.add_argument("--mpz-length-um", type=float, default=100.0)
    p.add_argument("--mpz-n-bins", type=int, default=200)
    p.add_argument("--mpz-source-bins", type=int, default=2)
    p.add_argument("--mpz-blunting-length-um", type=float, default=0.5)
    p.add_argument("--wake-length-um", type=float, default=100.0)
    p.add_argument("--wake-n-bins", type=int, default=0)
    p.add_argument("--wake-shielding", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wake-shield-projection", type=float, default=1.0)

    # Fatigue loading
    p.add_argument("--Kmax-MPa-sqrt-m", type=float, default=20.0, dest="Kmax_MPa_sqrt_m")
    p.add_argument("--R", type=float, default=0.1)
    p.add_argument("--frequency-Hz", type=float, default=1.0e3, dest="frequency_Hz")
    p.add_argument("--cycles-max", type=float, default=1.0e8, dest="cycles_max")
    p.add_argument("--block-cycles", type=float, default=1.0e4, dest="block_cycles")
    p.add_argument("--max-block-cycles", type=float, default=1.0e6, dest="max_block_cycles")
    p.add_argument("--min-block-cycles", type=float, default=1.0e-6, dest="min_block_cycles",
                   help="Minimum numerical cycle block. Fractional values are allowed so high-hazard cases fail within <1 cycle rather than being forced into an artificial one-cycle multi-fire jump.")
    p.add_argument("--no-adaptive-cycles", action="store_true")
    p.add_argument("--cycle-block-mode", choices=["requested_cap", "hazard_limited"], default="requested_cap", dest="cycle_block_mode",
                   help="requested_cap: --block-cycles is a hard upper bound. hazard_limited: use --max-block-cycles as the upper bound and choose the largest block allowed by fracture/plasticity hazard increments.")
    p.add_argument("--target-dB", type=float, default=0.2)
    p.add_argument("--target-dN-store", type=float, default=0.25, dest="target_dN_store")
    p.add_argument("--target-dN-emit", type=float, default=float("inf"), dest="target_dN_emit")
    p.add_argument("--target-dN-mobile", type=float, default=float("inf"), dest="target_dN_mobile")
    p.add_argument("--target-dN-escape", type=float, default=float("inf"), dest="target_dN_escape")
    p.add_argument("--target-dN-peierls", type=float, default=float("inf"), dest="target_dN_peierls")
    p.add_argument("--target-dN-taylor", type=float, default=float("inf"), dest="target_dN_taylor")
    p.add_argument("--n-phase", type=int, default=96, dest="n_phase")
    p.add_argument("--no-closure-clip", action="store_true")
    p.add_argument("--max-blocks", type=int, default=20000)
    p.add_argument("--n-advances", type=int, default=1)
    p.add_argument("--continue-after-fire", action="store_true")
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--no-plots", action="store_true", help="Disable automatic per-run and sweep plots.")
    p.add_argument("--sn-a0-m", type=float, default=None, help="Initial crack length for optional S-N conversion.")
    p.add_argument("--sn-Y", type=float, default=1.0, help="Geometry factor for optional S-N conversion.")

    # Sharp-front process-zone/fracture knobs
    p.add_argument("--r-pz", type=float, default=1.0e-6, dest="r_pz")
    p.add_argument("--sigma-cap-GPa", type=float, default=30.0, dest="sigma_cap_GPa")
    p.add_argument("--multihit-m", type=float, default=3.0, dest="multihit_m")
    p.add_argument("--multihit-tau", type=float, default=1.0e-6, dest="multihit_tau")
    p.add_argument("--nu0-cleave", type=float, default=1.0e12, dest="nu0_cleave")
    p.add_argument("--nu0-emit", type=float, default=1.0e11, dest="nu0_emit")
    p.add_argument("--dN-cap", type=float, default=float("inf"), dest="dN_cap",
                   help="Optional cap on legacy sharp-front emission increments. Default inf disables it; V1 fatigue plasticity uses adaptive cycle blocks instead.")
    p.add_argument("--beta-back", type=float, default=1.0, dest="beta_back")
    p.add_argument("--c-blunt", type=float, default=1.0, dest="c_blunt")
    p.add_argument("--L-pz", type=float, default=1.0e-6, dest="L_pz")
    p.add_argument("--v-emb-b3", type=float, default=500.0, dest="v_emb_b3")
    p.add_argument("--wake-retain", type=float, default=0.3, dest="wake_retain")
    p.add_argument("--cleave-shield-chi", type=float, default=0.0, dest="cleave_shield_chi")
    p.add_argument("--emb-sat-frac", type=float, default=1.0, dest="emb_sat_frac")
    p.add_argument("--n-sat", type=float, default=float("inf"), dest="N_sat")
    p.add_argument("--recover-k", type=float, default=0.0, dest="recover_k")
    p.add_argument("--rho0", type=float, default=5.0e12)
    p.add_argument("--da", type=float, default=2.0e-5)
    p.add_argument("--cleave-H0-eV", type=float, default=None, dest="cleave_H0_eV")
    p.add_argument("--cleave-S-sigma-max-kB", type=float, default=None, dest="cleave_S_sigma_max_kB")
    p.add_argument("--cleave-entropy-form", choices=["affine", "gated", "physical", "meyer_neldel"], default=None)
    p.add_argument("--cleave-barrier-kind", choices=["classic", "exp_floor"], default=None, dest="cleave_barrier_kind")
    p.add_argument("--cleave-G00-eV", type=float, default=None, dest="cleave_G00_eV")
    p.add_argument("--cleave-gT-eV-per-K", type=float, default=None, dest="cleave_gT_eV_per_K")
    p.add_argument("--cleave-sigc0-GPa", type=float, default=None, dest="cleave_sigc0_GPa")
    p.add_argument("--cleave-sT-GPa-per-K", type=float, default=None, dest="cleave_sT_GPa_per_K")
    p.add_argument("--cleave-exp-a", type=float, default=None, dest="cleave_exp_a")
    p.add_argument("--cleave-exp-n", type=float, default=None, dest="cleave_exp_n")
    p.add_argument("--cleave-floor-frac", type=float, default=None, dest="cleave_floor_frac")
    p.add_argument("--cleave-floor-min-eV", type=float, default=None, dest="cleave_floor_min_eV")
    p.add_argument("--cleave-floor-max-frac", type=float, default=None, dest="cleave_floor_max_frac")
    p.add_argument("--cleave-Tref-K", type=float, default=None, dest="cleave_Tref_K")
    p.add_argument("--cleave-exp-T-mode", choices=["linear", "mu_scale"], default=None, dest="cleave_exp_T_mode")
    p.add_argument("--cleave-mu-dlnmu-dT-per-K", type=float, default=None, dest="cleave_mu_dlnmu_dT_per_K")
    p.add_argument("--cleave-G0-mu-power", type=float, default=None, dest="cleave_G0_mu_power")
    p.add_argument("--cleave-sigc-mu-power", type=float, default=None, dest="cleave_sigc_mu_power")
    p.add_argument("--cleave-S-hs-kB", type=float, default=None, dest="cleave_S_hs_kB")
    p.add_argument("--cleave-sigma-S-GPa", type=float, default=None, dest="cleave_sigma_S_GPa")
    p.add_argument("--cleave-S-hs-power", type=float, default=None, dest="cleave_S_hs_power")
    p.add_argument("--cleave-S-hs-dT-per-K", type=float, default=None, dest="cleave_S_hs_dT_per_K")
    p.add_argument("--cleave-S-hs-Tref-K", type=float, default=None, dest="cleave_S_hs_Tref_K")
    p.add_argument("--cleave-monotone-stress", action=argparse.BooleanOptionalAction, default=None, dest="cleave_monotone_stress")

    # EXP-floor plasticity family and mechanism scalings
    p.add_argument("--exp-system", default="W[100]",
                   choices=["W[100]", "Ta[111]", "Al0.7CoCrFeNi-BCC", "Al0.7CoCrFeNi-FCC", "Cu"])
    p.add_argument("--exp-a", type=float, default=None, help="Override EXP-floor shape a.")
    p.add_argument("--exp-n", type=float, default=None, help="Override EXP-floor shape n.")
    p.add_argument("--nu0-emit-pz", type=float, default=1.0e11)
    p.add_argument("--nu0-peierls", type=float, default=1.0e12)
    p.add_argument("--nu0-taylor", type=float, default=1.0e11)
    p.add_argument("--emit-energy-scale", type=float, default=1.0)
    p.add_argument("--emit-entropy-scale", type=float, default=1.0)
    p.add_argument("--emit-stress-scale", type=float, default=1.0)
    p.add_argument("--peierls-energy-scale", type=float, default=0.02)
    p.add_argument("--peierls-entropy-scale", type=float, default=0.02)
    p.add_argument("--peierls-stress-scale", type=float, default=1.0)
    p.add_argument("--taylor-energy-scale", type=float, default=0.10)
    p.add_argument("--taylor-entropy-scale", type=float, default=0.10)
    p.add_argument("--taylor-stress-scale", type=float, default=1.0)

    # Process-zone storage/recovery
    p.add_argument("--storage-model", choices=["escape_limited", "all_retained", "fixed_fraction"],
                   default="escape_limited")
    p.add_argument("--fixed-retained-fraction", type=float, default=1.0)
    p.add_argument("--pz-recovery-per-s", type=float, default=0.0)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    summaries = [run_one_temperature(args, T) for T in args.temperatures]
    with open(os.path.join(args.out, "summary_all.json"), "w") as fp:
        json.dump(summaries, fp, indent=2, sort_keys=True)
    if not getattr(args, "no_plots", False):
        try:
            from .fatigue_postprocess import summarize_v1_sweep
            summarize_v1_sweep(args.out, a0_m=args.sn_a0_m, Y=args.sn_Y)
        except Exception as exc:
            print(f"  WARNING: sweep plotting failed for {args.out}: {exc}")
    print("\nSummary:")
    for s in summaries:
        print(
            f"T={s['T_K']:g}K cycles={s['cycles_total']:.3e} "
            f"n_adv={s['n_adv']} a={s['a_adv_m']*1e6:.3f}um "
            f"N_em={s['N_em']:.3g} B={s['B']:.3g}"
        )


if __name__ == "__main__":
    main()
