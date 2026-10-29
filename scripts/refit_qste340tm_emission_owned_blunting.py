#!/usr/bin/env python3
"""Joint QSTE340TM refit with emission-owned, moving-frame blunting.

This is an additive descendant of the accepted opening-reference calculation.
The sloping-floor opening shape is fixed to that reference.  One global floor
offset and shared EXP-floor plasticity/blunting coordinates are fitted to all
four R histories; no R-specific multiplier or offset is present.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.material_manifest import KB_EV_PER_K
from scripts.calibrate_qste340tm_opening_reference import (
    ATTEMPT_FREQUENCY_S,
    COOPERATIVE_HITS,
    COOPERATIVE_TAU_S,
    EVENT_LENGTH_M,
    NOMINAL_FREQUENCY_HZ,
    SMOOTHING_RATIO,
)
from scripts.fatigue_informed_joint_barrier_v3_v10230 import SmoothBroadBasis
from scripts.qste340tm_emission_blunting_reduction import (
    LOCAL_FORWARD_SLIP,
    LOCAL_SOURCE_SLIP,
    MOBILE,
    PROCESS_ZONE_M,
    RETAINED,
    WAKE_MOBILE,
    WAKE_RETAINED,
    BluntingParameters,
    advance_interval,
    effective_radius,
    initial_state,
    population_backstress,
    resolve_increment_advection,
    stress_cycle,
    wake_tip_projection,
)
from scripts.qste340tm_faithful_reduction import PlasticParameters, embrittlement_eV
from scripts.refit_qste340tm_restored_native import build_records


DATA = ROOT / "literature_data/fatigue_recovered_review"
OUT = DATA / "model_conditional/emission_owned_blunting"
REFERENCE = DATA / "model_conditional/conditional_opening_reference.json"
TEMPERATURE_K = 298.15
R_VALUES = (0.1, 0.3, 0.5, 0.7)
COLORS = {0.1: "#0072B2", 0.3: "#E69F00", 0.5: "#009E73", 0.7: "#CC79A7"}

# floor, cleavage-backstress fraction, emission G0, log sigc[GPa], transport G0,
# log recovery[s-1], log1p mobile0, log1p retained0, log source-blunt,
# log forward-blunt, log blunting-length[um]
LOWER = np.array([
    8.0, 0.0, 0.30, math.log(0.10), 0.30, math.log(1.0e-8),
    0.0, 0.0, math.log(1.0e-3), math.log(1.0e-3), math.log(0.10),
])
UPPER = np.array([
    25.0, 2.0, 4.0, math.log(30.0), 5.0, math.log(1.0e4),
    math.log1p(5000.0), math.log1p(5000.0), math.log(1.0e3),
    math.log(1.0e3), math.log(20.0),
])


def opening_reference() -> dict[str, float]:
    nominal = json.loads(REFERENCE.read_text())["nominal_fit"]["parameters"]
    return {
        "amplitude": float(nominal["g_amplitude_Gstar_over_kBT"]),
        "sigma_b_Pa": float(nominal["sigma_b_Pa"]),
        "exponent": float(nominal["exponent"]),
    }


def unpack(vector: np.ndarray) -> tuple[dict[str, float], PlasticParameters, BluntingParameters]:
    values = np.asarray(vector, dtype=float)
    scalar = {
        "opening_floor_over_kBT": float(values[0]),
        "cleavage_backstress_fraction": float(values[1]),
    }
    plastic = PlasticParameters(
        emission_G0_298_eV=float(values[2]),
        plastic_sigc_298_Pa=float(math.exp(values[3]) * 1.0e9),
        transport_G0_298_eV=float(values[4]),
        retained_recovery_per_s=float(math.exp(values[5])),
        initial_mobile_count=float(math.expm1(values[6])),
        initial_retained_count=float(math.expm1(values[7])),
        initial_source_activity=1.0,
        exp_floor_n=1.0,
    )
    blunting = BluntingParameters(
        source_slip_coefficient=float(math.exp(values[8])),
        forward_slip_coefficient=float(math.exp(values[9])),
        blunting_length_m=float(math.exp(values[10]) * 1.0e-6),
    )
    return scalar, plastic, blunting


def parameter_dict(vector: np.ndarray) -> dict[str, float]:
    scalar, plastic, blunting = unpack(vector)
    return scalar | {
        "emission_G0_298_eV": plastic.emission_G0_298_eV,
        "plastic_sigc_298_Pa": plastic.plastic_sigc_298_Pa,
        "transport_G0_298_eV": plastic.transport_G0_298_eV,
        "retained_recovery_per_s": plastic.retained_recovery_per_s,
        "initial_mobile_count": plastic.initial_mobile_count,
        "initial_retained_count": plastic.initial_retained_count,
        "source_slip_blunting_coefficient": blunting.source_slip_coefficient,
        "forward_slip_blunting_coefficient": blunting.forward_slip_coefficient,
        "blunting_length_m": blunting.blunting_length_m,
    }


def pack_parameters(parameters: dict[str, float]) -> np.ndarray:
    return np.asarray([
        parameters["opening_floor_over_kBT"],
        parameters["cleavage_backstress_fraction"],
        parameters["emission_G0_298_eV"],
        math.log(parameters["plastic_sigc_298_Pa"] / 1.0e9),
        parameters["transport_G0_298_eV"],
        math.log(parameters["retained_recovery_per_s"]),
        math.log1p(parameters["initial_mobile_count"]),
        math.log1p(parameters["initial_retained_count"]),
        math.log(parameters["source_slip_blunting_coefficient"]),
        math.log(parameters["forward_slip_blunting_coefficient"]),
        math.log(parameters["blunting_length_m"] / 1.0e-6),
    ], dtype=float)


def select_support(records: list[dict], points_per_r: int) -> list[dict]:
    selected: list[dict] = []
    for r_value in R_VALUES:
        block = [row for row in records if float(row["R"]) == r_value]
        positions = np.unique(
            np.linspace(0, len(block) - 1, min(points_per_r, len(block))).round().astype(int)
        )
        selected.extend(block[int(position)] for position in positions)
    return selected


def trajectory(
    vector: np.ndarray,
    records: list[dict],
    *,
    phases: int,
    radius_fraction_per_chunk: float = 0.08,
    maximum_chunk_cycles: int = 100_000,
    premeasurement_exposure_fraction: float = 0.0,
    resolved_population_advection: bool = False,
    advection_step_m: float | None = None,
) -> dict[str, np.ndarray]:
    _, plastic, blunting = unpack(vector)
    names = (
        "radius_m", "backstress_Pa", "mobile", "retained",
        "local_source_slip", "local_forward_slip", "adaptive_chunks",
        "wake_mobile", "wake_retained", "active_to_wake_mobile",
        "active_to_wake_retained", "wake_mobile_discarded",
        "wake_retained_discarded", "advection_substeps",
    )
    output = {name: np.zeros(len(records)) for name in names}
    output["stress"] = np.zeros((len(records), int(phases)))
    groups = {r: [i for i, row in enumerate(records) if float(row["R"]) == r] for r in R_VALUES}
    for r_value, indices in groups.items():
        state = initial_state(plastic)
        prior_cycles = float(records[indices[0]]["cycles"])
        prior_crack = float(records[indices[0]]["crack_length_m"])
        prior_dk = float(records[indices[0]]["DeltaK_MPa_sqrt_m"])
        for position, index in enumerate(indices):
            row = records[index]
            cycles = float(row["cycles"])
            crack = float(row["crack_length_m"])
            dk = float(row["DeltaK_MPa_sqrt_m"])
            diagnostic = {"chunks": 0.0}
            if position:
                state, diagnostic = advance_interval(
                    0.5 * (prior_dk + dk),
                    r_value,
                    state,
                    max(cycles - prior_cycles, 0.0),
                    max(crack - prior_crack, 0.0),
                    plastic,
                    blunting,
                    n_phase=phases,
                    radius_fraction_per_chunk=radius_fraction_per_chunk,
                    maximum_chunk_cycles=maximum_chunk_cycles,
                    resolved_population_advection=resolved_population_advection,
                    advection_step_m=advection_step_m,
                )
            elif premeasurement_exposure_fraction > 0.0:
                # The exact pre-measurement load/crack path is unavailable.
                # Applying the first measured load for the reported prior-cycle
                # count is an explicit upper exposure bracket, never a hidden
                # reconstruction of that missing protocol.
                state, diagnostic = advance_interval(
                    dk,
                    r_value,
                    state,
                    cycles * premeasurement_exposure_fraction,
                    0.0,
                    plastic,
                    blunting,
                    n_phase=phases,
                    radius_fraction_per_chunk=radius_fraction_per_chunk,
                    maximum_chunk_cycles=maximum_chunk_cycles,
                    resolved_population_advection=resolved_population_advection,
                    advection_step_m=advection_step_m,
                )
            stress, radius, backstress = stress_cycle(
                dk, r_value, state, blunting, phases
            )
            output["stress"][index] = stress
            output["radius_m"][index] = radius
            output["backstress_Pa"][index] = backstress
            output["mobile"][index] = state[MOBILE]
            output["retained"][index] = state[RETAINED]
            output["local_source_slip"][index] = state[LOCAL_SOURCE_SLIP]
            output["local_forward_slip"][index] = state[LOCAL_FORWARD_SLIP]
            output["wake_mobile"][index] = state[WAKE_MOBILE]
            output["wake_retained"][index] = state[WAKE_RETAINED]
            for key in (
                "active_to_wake_mobile", "active_to_wake_retained",
                "wake_mobile_discarded", "wake_retained_discarded",
                "advection_substeps",
            ):
                output[key][index] = float(diagnostic.get(key, 0.0))
            output["adaptive_chunks"][index] = diagnostic["chunks"]
            prior_cycles, prior_crack, prior_dk = cycles, crack, dk
    return output


def predict(vector: np.ndarray, state: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    scalar, _, _ = unpack(vector)
    reference = opening_reference()
    basis = SmoothBroadBasis(
        reference["sigma_b_Pa"], reference["exponent"], SMOOTHING_RATIO
    )
    opening_stress = np.maximum(
        state["stress"]
        - scalar["cleavage_backstress_fraction"] * state["backstress_Pa"][:, None],
        0.0,
    )
    raw = scalar["opening_floor_over_kBT"] + reference["amplitude"] * basis.value(
        opening_stress
    )
    embrittlement = np.asarray([embrittlement_eV(value) for value in state["retained"]])
    raw -= embrittlement[:, None] / (KB_EV_PER_K * TEMPERATURE_K)
    elementary = ATTEMPT_FREQUENCY_S * np.exp(-np.clip(raw, 0.0, 700.0))
    completed = gammainc(
        COOPERATIVE_HITS,
        np.minimum(elementary * COOPERATIVE_TAU_S, 1.0e12),
    ) / COOPERATIVE_TAU_S
    growth = EVENT_LENGTH_M * np.mean(completed, axis=1) / NOMINAL_FREQUENCY_HZ
    return np.maximum(growth, 1.0e-300), raw


def metrics(records: list[dict], prediction: np.ndarray, state: dict[str, np.ndarray]) -> dict:
    observed = np.asarray([float(row["observed_dadn_m_per_cycle"]) for row in records])
    ratios = np.asarray([float(row["R"]) for row in records])
    delta_k = np.asarray([float(row["DeltaK_MPa_sqrt_m"]) for row in records])
    error = np.log10(prediction / observed)
    blocks = {}
    for r_value in R_VALUES:
        mask = ratios == r_value
        blocks[str(r_value)] = {
            "rmse_log10_rate": float(np.sqrt(np.mean(error[mask] ** 2))),
            "bias_log10_rate": float(np.mean(error[mask])),
            "observed_slope": float(np.polyfit(np.log10(delta_k[mask]), np.log10(observed[mask]), 1)[0]),
            "predicted_slope": float(np.polyfit(np.log10(delta_k[mask]), np.log10(prediction[mask]), 1)[0]),
            "radius_range_m": [float(np.min(state["radius_m"][mask])), float(np.max(state["radius_m"][mask]))],
            "source_slip_final": float(state["local_source_slip"][mask][-1]),
            "forward_slip_final": float(state["local_forward_slip"][mask][-1]),
        }
    return {
        "overall_rmse_log10_rate": float(np.sqrt(np.mean(error**2))),
        "overall_bias_log10_rate": float(np.mean(error)),
        "maximum_radius_m": float(np.max(state["radius_m"])),
        "maximum_backstress_Pa": float(np.max(state["backstress_Pa"])),
        "maximum_population_density_m-2": float(
            np.max(state["mobile"] + state["retained"] + state["wake_mobile"] + state["wake_retained"]) / 1.0e-12
        ),
        "blocks": blocks,
    }


class Residual:
    def __init__(self, records: list[dict], phases: int, premeasurement_exposure_fraction: float,
                 resolved_population_advection: bool = False, advection_step_m: float | None = None):
        self.records = records
        self.phases = int(phases)
        self.premeasurement_exposure_fraction = float(premeasurement_exposure_fraction)
        self.resolved_population_advection = bool(resolved_population_advection)
        self.advection_step_m = advection_step_m
        self.observed = np.log10(
            [float(row["observed_dadn_m_per_cycle"]) for row in records]
        )
        ratios = np.asarray([float(row["R"]) for row in records])
        self.weights = np.empty(len(records))
        for r_value in R_VALUES:
            self.weights[ratios == r_value] = 1.0 / math.sqrt(np.sum(ratios == r_value))

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        try:
            # Acquisition uses bounded coarse feedback updates.  The selected
            # vector is replayed below with the stricter final integration.
            state = trajectory(
                vector,
                self.records,
                phases=self.phases,
                radius_fraction_per_chunk=0.50,
                maximum_chunk_cycles=1_000_000,
                premeasurement_exposure_fraction=self.premeasurement_exposure_fraction,
                resolved_population_advection=self.resolved_population_advection,
                advection_step_m=self.advection_step_m,
            )
            prediction, raw = predict(vector, state)
            error = np.log10(prediction) - self.observed
            centered = error.copy()
            ratios = np.asarray([float(row["R"]) for row in self.records])
            for r_value in R_VALUES:
                mask = ratios == r_value
                centered[mask] -= np.mean(error[mask])
            validity = max(0.0, 0.25 - float(np.min(raw))) * 10.0
            radius_penalty = max(0.0, float(np.max(state["radius_m"])) / 25.0e-6 - 1.0) * 5.0
            density = float(np.max(state["mobile"] + state["retained"])) / 1.0e-12
            density_penalty = max(0.0, math.log10(max(density, 1.0)) - 18.0) * 5.0
            backstress_penalty = max(0.0, float(np.max(state["backstress_Pa"])) / 20.0e9 - 1.0) * 5.0
            return np.r_[
                0.5 * error * self.weights,
                centered * self.weights,
                validity,
                radius_penalty,
                density_penalty,
                backstress_penalty,
            ]
        except (FloatingPointError, OverflowError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return np.full(2 * len(self.records) + 4, 1.0e6)


def starts() -> list[tuple[str, np.ndarray]]:
    common = np.array([
        16.4870880862, 0.2, 1.2, math.log(2.0), 1.2, math.log(1.0e-2),
        math.log1p(10.0), math.log1p(10.0), math.log(1.0), math.log(1.0), math.log(1.0),
    ])
    output = [("native_unit_blunting", common)]
    for label, emission, transport, source_blunt, forward_blunt, length_um in (
        ("source_owned", 1.0, 1.3, 30.0, 1.0, 2.0),
        ("forward_plastic", 1.0, 0.9, 3.0, 60.0, 3.0),
        ("mixed_blunting", 0.8, 1.1, 20.0, 20.0, 5.0),
        ("slow_transport", 1.0, 2.0, 40.0, 5.0, 5.0),
    ):
        vector = common.copy()
        vector[2] = emission
        vector[4] = transport
        vector[8] = math.log(source_blunt)
        vector[9] = math.log(forward_blunt)
        vector[10] = math.log(length_um)
        output.append((label, vector))
    return output


def write_csv(path: Path, records: list[dict], prediction: np.ndarray, state: dict[str, np.ndarray]) -> None:
    fields = list(records[0]) + ["predicted_dadn_m_per_cycle", "residual_log10"] + [
        key for key in state if key != "stress"
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(records):
            observed = float(row["observed_dadn_m_per_cycle"])
            extra = {
                "predicted_dadn_m_per_cycle": prediction[index],
                "residual_log10": math.log10(prediction[index] / observed),
            }
            extra.update({key: value[index] for key, value in state.items() if key != "stress"})
            writer.writerow(dict(row) | extra)


def plot(records: list[dict], prediction: np.ndarray, state: dict[str, np.ndarray]) -> None:
    ratios = np.asarray([float(row["R"]) for row in records])
    delta_k = np.asarray([float(row["DeltaK_MPa_sqrt_m"]) for row in records])
    observed = np.asarray([float(row["observed_dadn_m_per_cycle"]) for row in records])
    figure, axis = plt.subplots(figsize=(9.2, 6.3))
    for r_value in R_VALUES:
        mask = ratios == r_value
        order = np.argsort(delta_k[mask])
        color = COLORS[r_value]
        axis.plot(delta_k[mask][order], observed[mask][order], color=color, lw=2.1, label=f"R={r_value:.1f} experiment")
        axis.scatter(delta_k[mask], prediction[mask], facecolors="white", edgecolors=color, s=22, linewidths=1.0, label=f"R={r_value:.1f} fit")
    axis.set(xscale="log", yscale="log", xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)", ylabel=r"$da/dN$ (m/cycle)")
    axis.set_title("QSTE340TM resolved-increment 1-D refit")
    axis.grid(False)
    axis.legend(ncol=2, frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "combined_data_fit.png", dpi=240, bbox_inches="tight")
    figure.savefig(OUT / "combined_data_fit.svg", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    for r_value in R_VALUES:
        mask = ratios == r_value
        axis.plot(delta_k[mask], state["radius_m"][mask] * 1.0e6, color=COLORS[r_value], lw=2, label=f"R={r_value:.1f}")
    axis.set(xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)", ylabel=r"effective tip radius ($\mu$m)", title="Emission/forward-plasticity blunting history")
    axis.grid(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUT / "blunting_histories.png", dpi=240, bbox_inches="tight")
    figure.savefig(OUT / "blunting_histories.svg", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), sharex=True)
    for r_value in R_VALUES:
        mask = ratios == r_value
        order = np.argsort(delta_k[mask])
        color = COLORS[r_value]
        axes[0].plot(delta_k[mask][order], state["retained"][mask][order], color=color, lw=2, label=f"R={r_value:.1f}")
        axes[1].plot(delta_k[mask][order], state["wake_retained"][mask][order], color=color, lw=2)
    axes[0].set(ylabel="active retained count", yscale="symlog", title="Resolved moving-tip population histories")
    axes[1].set(xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)", ylabel="wake retained count", yscale="symlog")
    axes[0].legend(frameon=False, ncol=4)
    for axis in axes:
        axis.grid(False)
    figure.tight_layout()
    figure.savefig(OUT / "active_wake_population_histories.png", dpi=240, bbox_inches="tight")
    figure.savefig(OUT / "active_wake_population_histories.svg", bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-per-R", type=int, default=16)
    parser.add_argument("--fit-phases", type=int, default=8)
    parser.add_argument("--final-phases", type=int, default=32)
    parser.add_argument("--max-nfev", type=int, default=30)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--premeasurement-exposure-fraction", type=float, default=0.0)
    parser.add_argument("--resolved-population-advection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--advection-step-um", type=float, default=0.25)
    args = parser.parse_args(argv)
    if not 0.0 <= args.premeasurement_exposure_fraction <= 1.0:
        raise ValueError("premeasurement exposure fraction must lie in [0, 1]")
    global OUT
    if args.resolved_population_advection:
        OUT = OUT / "resolved_increment_1d"
    if args.premeasurement_exposure_fraction > 0.0:
        suffix = f"_premeasurement_{args.premeasurement_exposure_fraction:g}".replace(".", "p")
        OUT = OUT.with_name(OUT.name + suffix)
    OUT.mkdir(parents=True, exist_ok=True)
    records = build_records("index_rank", premeasurement=True)
    support = select_support(records, args.support_per_R)
    advection_step_m = args.advection_step_um * 1.0e-6
    residual = Residual(
        support, args.fit_phases, args.premeasurement_exposure_fraction,
        args.resolved_population_advection, advection_step_m,
    )
    initial_vectors = starts()
    if args.resume_from is not None:
        resumed = json.loads(args.resume_from.read_text())
        initial_vectors = [("resumed_selected", pack_parameters(resumed["parameters"]))]
    fits = []
    for label, initial in initial_vectors:
        solution = least_squares(
            residual,
            np.clip(initial, LOWER, UPPER),
            bounds=(LOWER, UPPER),
            max_nfev=args.max_nfev,
            x_scale="jac",
            ftol=2.0e-6,
            xtol=2.0e-6,
            gtol=2.0e-6,
        )
        score = float(np.dot(residual(solution.x), residual(solution.x)))
        fits.append((score, label, solution))
        print(f"[{label}] weighted_sse={score:.8g} nfev={solution.nfev}", flush=True)
    score, label, solution = min(fits, key=lambda item: item[0])
    state = trajectory(
        solution.x,
        records,
        phases=args.final_phases,
        premeasurement_exposure_fraction=args.premeasurement_exposure_fraction,
        resolved_population_advection=args.resolved_population_advection,
        advection_step_m=advection_step_m,
    )
    prediction, raw = predict(solution.x, state)
    result_metrics = metrics(records, prediction, state)
    _, selected_plastic, selected_blunting = unpack(solution.x)
    calibration_before = initial_state(selected_plastic)
    calibration_after, calibration_receipt = resolve_increment_advection(
        calibration_before, EVENT_LENGTH_M, selected_blunting,
        maximum_step_m=advection_step_m,
    )
    first_increment_operator = {
        "distance_m": EVENT_LENGTH_M,
        "maximum_subincrement_m": advection_step_m,
        "process_zone_length_m": PROCESS_ZONE_M,
        "wake_and_blunting_length_m": selected_blunting.blunting_length_m,
        "wake_tip_projection": wake_tip_projection(selected_blunting),
        "initial_state": calibration_before.tolist(),
        "translated_state": calibration_after.tolist(),
        "receipt": calibration_receipt,
        "reuse_policy": "same material-specific geometric generator applied to subsequent coarse crack increments",
    }
    payload = {
        "schema": "qste340tm-emission-owned-blunting/1",
        "status": "executed_state_informed_analytical_refit",
        "selected_start": label,
        "support_points": len(support),
        "fit_phases": args.fit_phases,
        "final_phases": args.final_phases,
        "premeasurement_exposure_fraction": args.premeasurement_exposure_fraction,
        "resolved_population_advection": args.resolved_population_advection,
        "advection_step_m": advection_step_m,
        "first_increment_operator": first_increment_operator,
        "premeasurement_exposure_policy": (
            "none; lower-bound sharp state at first observation"
            if args.premeasurement_exposure_fraction == 0.0
            else "declared fraction of reported prior cycles at first measured load; upper-bracket approximation with zero inferred prehistory crack advance"
        ),
        "parameters": parameter_dict(solution.x),
        "fixed_opening_shape": opening_reference(),
        "metrics": result_metrics,
        "raw_barrier_over_kBT_range": [float(np.min(raw)), float(np.max(raw))],
        "optimizer": {
            "success": bool(solution.success), "status": int(solution.status),
            "message": str(solution.message), "nfev": int(solution.nfev),
            "weighted_sse": score,
        },
        "multi_start": [
            {"start": item_label, "weighted_sse": item_score, "nfev": int(item_solution.nfev), "parameters": parameter_dict(item_solution.x)}
            for item_score, item_label, item_solution in sorted(fits)
        ],
        "ownership": {
            "blunting": "local gross source slip plus forward plastic passage, convected by crack advance",
            "backstress_and_stored_energy": "current mobile/retained population only",
            "R_specific_fit_parameters": False,
            "population_advection": (
                "resolved active-to-wake transaction using process-zone and blunting lengths; no fitted advection fraction"
                if args.resolved_population_advection else "historical tip-attached population"
            ),
        },
        "limitations": [
            "state-informed analytical reduction, not a replacement for native spatial qualification",
            "frequency remains the user-authorized provisional 10 Hz",
            "source-order crack-history alignment is approximate",
            "finite multi-start fit does not establish parameter uniqueness",
        ],
    }
    (OUT / "fit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(OUT / "predictions.csv", records, prediction, state)
    plot(records, prediction, state)
    print(json.dumps({"output": str(OUT), "metrics": result_metrics}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
