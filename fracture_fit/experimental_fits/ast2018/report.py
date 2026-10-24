"""Fail-closed final Ast fit report assembled from complete run artifacts."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import AstData
from .config import AstAssumptions
from .native import historical_start,load_native
from .objective import PARAMETERS,material_from_normalized
from .provenance import forward_adapter_sha256
from .selection import retained_vector_path
from .runtime import account


SCENARIOS=("TABLE_PRIMARY","FIGURE5_ALTERNATIVE")
PROFILED_FITTED_KEYS={PARAMETERS[i][0] for i in (0,1,3,6)}
PROFILED_FIXED_KEYS={"rho_source0_m2","physics.r0_m","physics.mpz_length_m",
                     "physics.source_zone_length_m","physics.cleavage_nu0_s",
                     "physics.emission_nu0_s"}


def _unit(field:str)->str:
    if field in ("rho_source0_m2","taylor_corr_rho_c_m2"):
        return "m^-2"
    if field.endswith("_eV_per_K"):return "eV/K"
    if field.endswith("_Pa_per_K"):return "Pa/K"
    if field.endswith("_eV"):return "eV"
    if field.endswith("_Pa"):return "Pa"
    if field.endswith("_K"):return "K"
    if field=="nu0_s":return "s^-1"
    if field=="activation_entropy_kB":return "kB"
    return "dimensionless"


def _load(path:Path):
    if not path.exists():raise RuntimeError(f"missing required final artifact: {path}")
    return json.loads(path.read_text())


def _parameter_rows(source:Path,vector:Path,scenario:str,stage:str,
                    assumptions:AstAssumptions,reference_condition:dict):
    mods,_=load_native(source)
    base,physics,_=historical_start(source,mods[2],mods[3].CommonPhysics)
    x=_load(vector)["x"]
    candidate=material_from_normalized(base,x)
    physics=assumptions.physics(physics,assumptions.specimen(reference_condition))
    fields=asdict(candidate)
    rows=[]
    for key,lo,hi in PARAMETERS:
        group,field=key.split(".")
        value=fields[group][field]
        rows.append({"scenario":scenario,"stage":stage,"code_key":key,
                     "value":value,"unit":_unit(field),
                     "lower":lo,"upper":hi,"status":"fitted",
                     "profile_status":"conditional profile" if key in PROFILED_FITTED_KEYS
                                      else "unprofiled fitted coordinate"})
    fitted={key for key,_,_ in PARAMETERS}
    for group in ("cleavage","emission","peierls","taylor"):
        for field,value in fields[group].items():
            key=f"{group}.{field}"
            if key in fitted:continue
            rows.append({"scenario":scenario,"stage":stage,"code_key":key,
                         "value":value,"unit":_unit(field),
                         "lower":"","upper":"","status":"fixed",
                         "profile_status":"not profiled"})
    for field in ("rho_source0_m2","taylor_corr_rho_c_m2","taylor_corr_scale","c_blunt"):
        rows.append({"scenario":scenario,"stage":stage,"code_key":field,
                     "value":fields[field],"unit":_unit(field),"lower":"","upper":"",
                     "status":"fixed","profile_status":
                         "scenario sensitivity" if field in PROFILED_FIXED_KEYS else "not profiled"})
    for field,unit in (("cleavage_nu0_s","s^-1"),("emission_nu0_s","s^-1"),
                       ("cleavage_hits","count"),("cleavage_correlation_time_s","s"),
                       ("n_systems","count"),("source_zone_length_m","m"),
                       ("r0_m","m"),("mpz_length_m","m"),
                       ("b_m","m"),("n_bins","count"),
                       ("reference_source_area_m2","m^2"),
                       ("reference_density_m2","m^-2"),
                       ("rho_forest_floor_m2","m^-2"),
                       ("reference_front_width_m","m"),
                       ("active_strip_width_m","m"),
                       ("activation_to_line_content_per_system","dimensionless tuple"),
                       ("emission_schmid_factors","dimensionless tuple"),
                       ("shielding_orientation_factors","dimensionless tuple"),
                       ("mobile_transport_velocity_scale","dimensionless")):
        rows.append({"scenario":scenario,"stage":stage,"code_key":f"physics.{field}",
                     "value":getattr(physics,field),"unit":unit,"lower":"","upper":"",
                     "status":"fixed","profile_status":
                         "scenario sensitivity" if f"physics.{field}" in PROFILED_FIXED_KEYS
                         else "not profiled"})
    return rows


def _onset_metrics(data,scenario,summaries):
    indexed={(r["level"],r["condition_id"]):r for r in summaries}
    out={}
    for level in ("reduced","spatial"):
        errors=[];supported=0;within=0
        for row in data.toughness_scenario(scenario):
            if row["observable"]!="Kq_Jcontinuous_onset":continue
            pred=indexed.get((level,row["condition_id"]),{}).get("onset_K_Jcontinuous_MPa_sqrt_m")
            if pred is None:continue
            target=float(row["value"])
            supported+=1;errors.append(pred-target)
            within+=abs(pred-target)<=max(.2,.05*target)
        out[level]={"n_supported":supported,"n_observed":23,
                    "n_within_working_tolerance":within,
                    "mean_absolute_error_MPa_sqrt_m":float(np.mean(np.abs(errors))) if errors else None,
                    "mean_signed_error_MPa_sqrt_m":float(np.mean(errors)) if errors else None}
    return out


def _annotate_local_identifiability(rows:list[dict],scenario:str,jacobian:np.ndarray):
    if jacobian.ndim!=2 or jacobian.shape[1]!=len(PARAMETERS):
        raise RuntimeError(f"invalid fitted Jacobian shape for {scenario}")
    norms=np.linalg.norm(jacobian,axis=0)
    relative_floor=1e-8*float(np.max(norms))
    flat=[]
    for index,(key,_,_) in enumerate(PARAMETERS):
        if not np.isfinite(norms[index]):
            raise RuntimeError(f"nonfinite local sensitivity for {scenario}: {key}")
        if norms[index]<=relative_floor:
            flat.append(key)
        for row in rows:
            if (row["scenario"]==scenario and row["stage"]=="analytical" and
                row["code_key"]==key):
                row["local_jacobian_column_norm"]=float(norms[index])
                if norms[index]<=relative_floor:
                    row["status"]="optimized; locally unidentifiable"
                    row["profile_status"]+=("; exact zero local finite-difference sensitivity"
                        if norms[index]==0 else
                        "; near-zero local finite-difference sensitivity relative to largest fitted column")
    return flat


def _curve_support_metrics(rows:list[dict],scenario:str):
    result={}
    for level in ("reduced","spatial"):
        selected=[row for row in rows if row["level"]==level]
        points=sum(int(row["all_points"]) for row in selected)
        covered=sum(int(row["covered_all_points"]) for row in selected)
        if len(selected)!=8 or points!=629 or covered<0 or covered>points:
            raise RuntimeError(f"incomplete full J support audit: {scenario} {level}")
        result[level]={"all_points":points,"covered_points":covered,
                       "all_point_cost":sum(float(row["all_629_support_cost"])
                                            for row in selected)}
    return result


def build_report(source:Path,data_dir:Path,run_dir:Path):
    data=AstData.load(data_dir)
    integrity=_load(run_dir/"input_integrity.json")
    environment=_load(run_dir/"environment_receipt.json")
    extracted_integrity=_load(run_dir/"extracted_integrity.json")
    if extracted_integrity["archive_sha256"]!=integrity["archive_sha256"] or \
       extracted_integrity["verified_extracted_file_count"]!=integrity["verified_file_count"]:
        raise RuntimeError("extracted data integrity receipt is incomplete")
    cfg={s:_load(run_dir/f"config_{s.lower()}.json") for s in SCENARIOS}
    if any(cfg[s]["archive_sha256"]!=integrity["archive_sha256"] for s in SCENARIOS):
        raise RuntimeError("fitted scenario archive hash differs from verified input")
    best={s:_load(run_dir/f"best_{s.lower()}.json") for s in SCENARIOS}
    searches={s:_load(run_dir/f"search_summary_{s.lower()}.json") for s in SCENARIOS}
    confirmation={s:_load(run_dir/f"confirmation_{s.lower()}/summary.json") for s in SCENARIOS}
    if any(len(confirmation[s])!=62 for s in SCENARIOS):
        raise RuntimeError("final paired confirmation must contain 31 conditions at each level")
    for s in SCENARIOS:
        forward_hash=forward_adapter_sha256()
        search=searches[s]
        if search["n_requested"]!=cfg[s]["starts"] or not search.get("best"):
            raise RuntimeError(f"incomplete bounded analytical search in {s}")
        priority=_load(run_dir/f"confirmation_{s.lower()}_priority11/summary.json")
        if len(priority)!=22 or any(r["status"].startswith("numerical") for r in priority):
            raise RuntimeError(f"incomplete unchanged-vector priority confirmation in {s}")
        priority_vector=_load(run_dir/f"confirmation_{s.lower()}_priority11/vector.json")
        if priority_vector["x"]!=best[s]["x"]:
            raise RuntimeError(f"priority confirmation is not at the unchanged analytical vector: {s}")
        final_vector=_load(run_dir/f"confirmation_{s.lower()}/vector.json")
        endpoint=_load(run_dir/f"confirmation_{s.lower()}_endpoint_validation/summary.json")
        endpoint_vector=_load(run_dir/f"confirmation_{s.lower()}_endpoint_validation/vector.json")
        if (len(endpoint)!=16 or not endpoint_vector["validation_endpoint_horizon"] or
            any(r["status"].startswith("numerical") for r in endpoint)):
            raise RuntimeError(f"incomplete endpoint-validation trajectories in {s}")
        retained=retained_vector_path(run_dir,s)
        retained_x=_load(retained)["x"]
        if endpoint_vector["x"]!=retained_x:
            raise RuntimeError(f"endpoint validation vector differs from final retained vector in {s}")
        if final_vector["x"]!=retained_x:
            raise RuntimeError(f"final confirmation vector differs from retained vector in {s}")
        if any(vector.get("forward_adapter_sha256")!=forward_hash for vector in
               (priority_vector,final_vector,endpoint_vector)):
            raise RuntimeError(f"paired forward adapter changed across confirmation stages: {s}")
        if retained.name.startswith("best_1d_refined_"):
            audit_stem=(f"spatial_local_runtime_audit_{s.lower()}_profile_seeded"
                        if retained.name.endswith("_profile_seeded.json") else
                        f"spatial_local_runtime_audit_{s.lower()}")
            for extension in ("png","svg"):
                path=run_dir/f"{audit_stem}.{extension}"
                if not path.exists():
                    raise RuntimeError(f"missing direct 1-D local runtime audit: {path}")
        observed_pairs={(row["level"],row["condition_id"]) for row in confirmation[s]}
        required_pairs={(level,row["condition_id"]) for level in ("reduced","spatial")
                        for row in data.conditions}
        if observed_pairs!=required_pairs:
            raise RuntimeError(f"missing or duplicate paired condition in {s}")
        numerical=[row for row in confirmation[s]
                   if row["status"].startswith("numerical") or row["numerical_failure"]]
        if numerical:
            raise RuntimeError(f"unresolved numerical confirmation failures in {s}: "
                               f"{[(r['level'],r['condition_id']) for r in numerical]}")
        for level,condition_id in required_pairs:
            folder=run_dir/f"confirmation_{s.lower()}/{level}/{condition_id}"
            for filename in ("mechanical_history.csv","ast_observer.csv",
                             "events.json","summary.json"):
                path=folder/filename
                if not path.exists() or path.stat().st_size==0:
                    raise RuntimeError(f"missing complete confirmation artifact: {path}")
        for artifact in (f"curve_support_{s.lower()}.json",
                         f"search_recovery_{s.lower()}.json",
                         f"search_runtime_audit_{s.lower()}.png",
                         f"search_runtime_audit_{s.lower()}.svg",
                         f"refine_runtime_audit_{s.lower()}.png",
                         f"refine_runtime_audit_{s.lower()}.svg",
                         f"convergence_{s.lower()}.json",
                         f"holdouts_{s.lower()}.json",
                         f"no_cleavage_outcomes_{s.lower()}.json",
                         f"sensitivity_{s.lower()}.json",
                         f"ensemble_{s.lower()}_reduced/summary.json",
                         f"ensemble_{s.lower()}_reduced/runs.json",
                         f"ensemble_{s.lower()}_spatial/summary.json",
                         f"ensemble_{s.lower()}_spatial/runs.json",
                         f"identifiability_{s.lower()}/sensitivity.json",
                         f"identifiability_{s.lower()}/scaled_jacobian.npz",
                         f"identifiability_{s.lower()}/profiles.json",
                         f"barriers_{s.lower()}/barrier_surfaces.csv",
                         f"barriers_{s.lower()}/barrier_visited_event_rates.csv",
                         f"barriers_{s.lower()}/barrier_sensitivity_envelope.csv",
                         f"barriers_{s.lower()}/barrier_receipt.json",
                         f"barriers_{s.lower()}/barrier_derivatives.png",
                         f"barriers_{s.lower()}/barrier_derivatives.svg",
                         f"barriers_{s.lower()}/barrier_rates.png",
                         f"barriers_{s.lower()}/barrier_rates.svg"):
            if not (run_dir/artifact).exists():raise RuntimeError(f"missing {artifact}")
        reduced_ensemble=_load(run_dir/f"ensemble_{s.lower()}_reduced/summary.json")
        spatial_ensemble=_load(run_dir/f"ensemble_{s.lower()}_spatial/summary.json")
        for level in ("reduced","spatial"):
            ensemble_basis=_load(run_dir/f"ensemble_{s.lower()}_{level}/checkpoint_basis.json")
            if ensemble_basis.get("forward_adapter_sha256")!=forward_hash or \
               ensemble_basis["vector"]!=retained_x:
                raise RuntimeError(f"ensemble does not match final forward adapter/vector: {s} {level}")
        if len(reduced_ensemble)!=31 or len(spatial_ensemble)!=11:
            raise RuntimeError(f"incomplete stochastic ensemble for {s}")
        figures=run_dir/f"figures_{s.lower()}"
        for name in ("onset_by_rate","cleavage_validation_only","J_resistance_curves",
                     "nominal_K_loading_traces","relative_stiffness","residual_blocks",
                     "A2_protocol_check","unchanged_transfer_onset",
                     "unchanged_transfer_J_curves",
                     "unchanged_transfer_native_state"):
            if not (figures/f"{name}.png").exists() or not (figures/f"{name}.svg").exists():
                raise RuntimeError(f"missing figure pair {name} for {s}")
    runtime=account(run_dir)
    residual_rows=[]
    comparison_rows=[]
    for scenario in SCENARIOS:
        indexed={(row["level"],row["condition_id"]):row
                 for row in confirmation[scenario]}
        for row in confirmation[scenario]:
            for block in row["residual_blocks"]:
                residual_rows.append({"scenario":scenario,"level":row["level"],
                    "condition_id":row["condition_id"],"status":row["status"],
                    "family":block["family"],"curve_id":block["curve_id"],
                    "working_cost":block["cost"],
                    "rms_standardized":block["rms_standardized"]})
        for observed in data.toughness_scenario(scenario):
            observable=observed["observable"]
            predicted_key={"Kq_Jcontinuous_onset":"onset_K_Jcontinuous_MPa_sqrt_m",
                           "Kq_LEFM":"LEFM_Kq_MPa_sqrt_m",
                           "Kq_Jbasic_cleavage":"basic_J_Kq_a0_MPa_sqrt_m"}.get(observable)
            for level in ("reduced","spatial"):
                prediction=indexed[(level,observed["condition_id"])]
                comparison_rows.append({"scenario":scenario,"level":level,
                    "measurement_id":observed["measurement_id"],
                    "condition_id":observed["condition_id"],
                    "observable":observable,"observed_MPa_sqrt_m":observed["value"],
                    "predicted_MPa_sqrt_m":prediction.get(predicted_key) if predicted_key else None,
                    "prediction_status":prediction["status"],
                    "relation":("same continuous-J onset definition" if
                                observable=="Kq_Jcontinuous_onset" else
                                "geometric ligament endpoint; not native unstable cleavage")})
    for filename,rows in (("residual_contributions.csv",residual_rows),
                          ("toughness_predictions.csv",comparison_rows)):
        with (run_dir/filename).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    source_rows=[]
    for level in ("reduced","spatial"):
        p={r["condition_id"]:r for r in confirmation["TABLE_PRIMARY"]
           if r["level"]==level}
        a={r["condition_id"]:r for r in confirmation["FIGURE5_ALTERNATIVE"]
           if r["level"]==level}
        observed_p={r["condition_id"]:float(r["value"]) for r in
                    data.toughness_scenario("TABLE_PRIMARY")
                    if r["observable"]=="Kq_Jcontinuous_onset"}
        observed_a={r["condition_id"]:float(r["value"]) for r in
                    data.toughness_scenario("FIGURE5_ALTERNATIVE")
                    if r["observable"]=="Kq_Jcontinuous_onset"}
        for condition_id in sorted(p):
            pv=p[condition_id]["onset_K_Jcontinuous_MPa_sqrt_m"]
            av=a[condition_id]["onset_K_Jcontinuous_MPa_sqrt_m"]
            source_rows.append({"level":level,"condition_id":condition_id,
                "table_primary_observed_onset_MPa_sqrt_m":observed_p.get(condition_id),
                "figure5_alternative_observed_onset_MPa_sqrt_m":observed_a.get(condition_id),
                "table_primary_predicted_onset_MPa_sqrt_m":pv,
                "figure5_alternative_predicted_onset_MPa_sqrt_m":av,
                "prediction_delta_alternative_minus_primary_MPa_sqrt_m":
                    av-pv if pv is not None and av is not None else None})
    with (run_dir/"source_scenario_comparison.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(source_rows[0]))
        writer.writeheader();writer.writerows(source_rows)
    params=[]
    for scenario in SCENARIOS:
        params.extend(_parameter_rows(source,run_dir/f"best_{scenario.lower()}.json",
                                      scenario,"analytical",
                                      AstAssumptions(**cfg[scenario]["ast_assumptions"]),
                                      data.conditions[0]))
        local=retained_vector_path(run_dir,scenario)
        if local.name.startswith("best_1d_refined_"):params.extend(_parameter_rows(
            source,local,scenario,"local_1d_refined",
            AstAssumptions(**cfg[scenario]["ast_assumptions"]),data.conditions[0]))
    for row in params:
        row["local_jacobian_column_norm"]=""
    flat_keys={}
    for scenario in SCENARIOS:
        jacobian=np.load(run_dir/f"identifiability_{scenario.lower()}/scaled_jacobian.npz")["jacobian"]
        flat_keys[scenario]=_annotate_local_identifiability(params,scenario,jacobian)
    parameter_file=run_dir/"fitted_parameter_table.csv"
    with parameter_file.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(params[0]));writer.writeheader();writer.writerows(params)
    onset={s:_onset_metrics(data,s,confirmation[s]) for s in SCENARIOS}
    j_support={s:_curve_support_metrics(_load(run_dir/f"curve_support_{s.lower()}.json"),s)
               for s in SCENARIOS}
    lines=["# Ast 2018 tungsten joint fit", "",
        "Experimental source: [Ast et al., Materials & Design 152 (2018) 168–180](https://doi.org/10.1016/j.matdes.2018.04.009). The supplied verified extraction, rather than a new digitization, is the numerical input.","",
        "## Source and executed model", "",
        f"- Native source: `{cfg['TABLE_PRIMARY']['source_receipt']['commit']}` at `{cfg['TABLE_PRIMARY']['source_receipt']['repository']}`.",
        f"- Frozen 0-D search adapter SHA-256: `{cfg['TABLE_PRIMARY']['adapter_sha256']}`; paired forward-wrapper SHA-256: `{forward_adapter_sha256()}`. Separate hashes expose spatial localization changes without reinterpreting a previously checkpointed 0-D search.",
        f"- Supplied extraction archive SHA-256: `{cfg['TABLE_PRIMARY']['archive_sha256']}`. Its internal SHA256SUMS verified {integrity['verified_file_count']} files (`input_integrity.json`).",
        "- The same seven active material coordinates are shared across temperatures, rates and fitted observable families. Shape exponents, EXP-floor fractions, stress thermal slopes, source density, attempt frequencies, Taylor correlation and blunting remain fixed under the stated historical starting convention; separate sensitivity runs probe selected assumptions.",
        "- Search candidates have a 900 s per-vector numerical watchdog. Every timeout is checkpointed as an invalid objective evaluation with its completed-condition trace; it cannot be interpreted as a material prediction. `search_runtime_audit_*` plots show completed costs and watchdog rejections.",
        "- The main search uses the stateful native v9.13 0-D reduction. Direct v9.13 spatial 1-D confirmations use `translation_mode='hazard_coupled'`, `translation_action_exponent=0.95`, and an explicit small-source integration threshold of zero.",
        "- No 2-D solver or historical campaign was executed.","",
        "## Mechanics and observation operator", "",
        "- Nominal `K=F κ(a0)` and current elastic `K=F κ(a)` use Ast's polynomial geometry factor. Compliance integrates `dC/da=2B κ²/E′` from the initial notch; the initial compliance is constrained by the A2 150 nm/s and initial 1.6 MPa√m/s protocol example, scales with known E(T), and is clipped to the reported 5–150 nm/s range. Exact specimen compliance, notch radius and instrument compliance were not supplied.",
        "- Scalar force equilibrium is `u=[C_e(a)+C_inst]F+u_pl`. The added plastic load-point displacement is an Orowan mobile-density/Peierls-velocity virtual-work projection over the declared MPZ. The 1-D projection uses the native internally integrated external plastic-work ledger and a source-zone average of the same Mode-I unit-force field. This is a new observation closure and remains an approximation to the full three-dimensional cantilever field.",
        "- Plastic area is integrated from the predicted force-displacement history minus the elastic unloading triangle. Continuous J uses Ast's incremental ligament correction and `Kq,J` is evaluated only after the trajectory reaches Δa=0.2 µm. LEFM 95%-line and basic-J results require a separately identified physical ligament endpoint; their values are validation checks, not terms in the fitted objective.",
        "- Event advance is a declared physical 20 nm coarse-graining length, with independent unit-exponential hazard thresholds in ensemble runs. The deterministic unit-action trajectory initializes the fit and is not interpreted as a mean event time. Geometry/protocol/event sensitivities and common seed provenance are in the machine-readable outputs.","",
        "## Working fit and source scenarios", "",
        "- The fitted mask contains 23 Table 1 continuous-J onset summaries, eight J resistance curves, supported Fig. 2 loading branches, and three relative-stiffness histories. Table 1 and Fig. 5 means are used once per source scenario. Digitized curve blocks use fixed quadrature and robust working tolerances; these costs are not a probability likelihood or confidence intervals.",
        "- `TABLE_PRIMARY` retains 7.6 and 5.8 MPa√m at 75 and 100 °C, 1.6 MPa√m/s. `FIGURE5_ALTERNATIVE` substitutes only the separately digitized 8.858 and 7.5652 MPa√m centers. The 75 °C cleavage summary remains distinct and unchanged.",
        "", "| Scenario | Analytical working cost | 0-D onset MAE | 1-D onset MAE | Supported 0-D / 1-D onsets | Within working tolerance, 0-D / 1-D |", "|---|---:|---:|---:|---:|---:|"]
    for s in SCENARIOS:
        a=onset[s];r=a["reduced"];q=a["spatial"]
        lines.append(f"| {s} | {best[s]['cost']:.4g} | {r['mean_absolute_error_MPa_sqrt_m']} | {q['mean_absolute_error_MPa_sqrt_m']} | {r['n_supported']} / {q['n_supported']} of 23 | {r['n_within_working_tolerance']} / {q['n_within_working_tolerance']} of 23 |")
    lines.append("\nThe onset count uses the fitted working tolerance `max(0.2, 0.05 × observed K)` MPa√m. Paired 0-D/1-D agreement measures transfer between the native approximations; experimental agreement is quantified by the separate onset counts and curve residuals. The two scenario costs use different supplied onset centers and are reported as source sensitivity, not as a likelihood ratio or source-selection probability.")
    for s in SCENARIOS:
        row=searches[s]
        lines.append(f"\n- {s} space-filling search: {row['n_valid']} valid completed objectives of {row['n_requested']} proposed vectors; {row.get('n_watchdog_rejections',0)} watchdog rejections. Inference is conditional on this numerical search coverage.")
    lines.extend(["", "| Active material key | TABLE_PRIMARY | FIGURE5_ALTERNATIVE |",
                  "|---|---:|---:|"])
    for index,(key,_,_) in enumerate(PARAMETERS):
        a=float(best["TABLE_PRIMARY"]["x"][index])
        b=float(best["FIGURE5_ALTERNATIVE"]["x"][index])
        lines.append(f"| `{key}` (normalized bound coordinate) | {a:.4f} | {b:.4f} |")
    bound_hits={s:[key for (key,_,_),value in zip(PARAMETERS,best[s]["x"])
                   if float(value)<.005 or float(value)>.995] for s in SCENARIOS}
    lines.append("\nBound hits within 0.5% of a fitted interval: "+
                 "; ".join(f"{s}: {', '.join(bound_hits[s]) or 'none'}" for s in SCENARIOS)+".")
    lines.extend(["", "These are working-scale optimization and prediction diagnostics. A large residual or missing onset is retained as model evidence.","",
        "## Identifiability and sensitivity",""])
    for s in SCENARIOS:
        sens=_load(run_dir/f"identifiability_{s.lower()}/sensitivity.json")
        values=sens["singular_values"]
        lines.append(f"- {s}: normalized sensitivity singular values: {', '.join(f'{v:.3g}' for v in values)}. Conditional profile costs and bound hits are in `identifiability_{s.lower()}/profiles.json`; these are sensitivity intervals, not statistical confidence intervals.")
        lines.append(f"- {s}: locally zero or below 1e-8 of the largest fitted Jacobian-column norm at the recorded normalized finite-difference step: {', '.join('`'+key+'`' for key in flat_keys[s]) or 'none'}. Such optimized coordinates are not independently identified by these working observables at this vector.")
        profiles=_load(run_dir/f"identifiability_{s.lower()}/profiles.json")
        grouped={}
        for profile in profiles:
            grouped.setdefault(profile["parameter_key"],[]).append(profile)
        for key,points in sorted(grouped.items()):
            costs=[float(point["cost"]) for point in points]
            offsets=[float(point["fixed_normalized"]) for point in points]
            lines.append(f"- {s} `{key}` conditional profile: normalized fixed points {', '.join(f'{v:.3f}' for v in offsets)}; reoptimized costs {', '.join(f'{v:.4g}' for v in costs)}. Cost span {max(costs)-min(costs):.4g} across the tested points.")
    lines.extend(["- Radius, specimen geometry, MPZ/source zone, attempt-frequency, source-density, physical event length and working-weight variations are recorded in each `sensitivity_*.json`. The fitted barriers are conditional on these assumptions.","",
        "- `holdouts_*.json` gives local conditional leave-one-condition-out training probes at a cold, warm and hot condition, with direct held-out costs. This is a sensitivity check around the selected fit rather than a global cross-validation refit.","",
        "## Direct 1-D transfer and residuals",""])
    for s in SCENARIOS:
        convergence=_load(run_dir/f"convergence_{s.lower()}.json")
        change=convergence["differences"]["middle_to_fine_cost"]
        lines.append(f"- {s}: tightening the full 0-D objective from 0.025 to 0.0125 MPa√m load steps and from 14 to 16 first-passage bisections changed the working cost by {change:.5g}; see `convergence_{s.lower()}.json` for all frozen residuals and call times.")
    for s in SCENARIOS:
        rows=confirmation[s]
        statuses={level:{} for level in ("reduced","spatial")}
        for row in rows:
            level=row["level"];status=row["status"]
            statuses[level][status]=statuses[level].get(status,0)+1
        lines.append(f"- {s}: 0-D statuses `{statuses['reduced']}`; 1-D statuses `{statuses['spatial']}`. Per-condition event, force, J, stiffness and native state diagnostics are in `confirmation_{s.lower()}/`.")
        for level in ("reduced","spatial"):
            support=j_support[s][level]
            lines.append(f"- {s} {level}: the direct prediction reaches {support['covered_points']} of {support['all_points']} supplied J-curve support points; full-point working cost is {support['all_point_cost']:.4g}. The uncovered points remain residual evidence outside the model trajectory's J range.")
        outcomes=_load(run_dir/f"no_cleavage_outcomes_{s.lower()}.json")
        for level in ("reduced","spatial"):
            selected=[row for row in outcomes if row["level"]==level]
            if len(selected)!=8:
                raise RuntimeError(f"incomplete no-cleavage endpoint audit: {s} {level}")
            survived=sum(row["check_status"]=="survived_through_observed_nominal_K"
                         for row in selected)
            lines.append(f"- {s} {level}: {survived} of 8 separately loaded no-cleavage endpoints were survived to their observed nominal K; this is a qualitative endpoint check.")
    lines.extend(["- `curve_support_*.json` compares the search's fixed 25-node curve quadrature, all 213 flagged fit-subset graphical samples, and all 629 J-curve samples without increasing a curve's weight through raster density. The eight full observed and predicted J curves, onset/cleavage panels, nominal-K traces, stiffness panels and residual blocks have PNG and SVG versions in `figures_*/`. `unchanged_transfer_*` figures use the analytical vector before any conditional 1-D refinement and include native activation, backstress and source-multiplicity histories.",
        "- `barriers_*/barrier_surfaces.csv` contains executed G*, H*, S*, V*, floor/cap flags and elementary rates over each mechanism's effective stress range visited at confirmed spatial events. Cleavage also has the effective multi-hit rate. `barrier_visited_event_rates.csv` records native state-dependent cleavage, source activation, mobile encounter and retained release rates at each event; carrier-weighted rates are undefined where the relevant carrier population is zero. The inherited sublinear emission stress exponent makes V* undefined exactly at zero stress; that row is flagged rather than assigned a finite volume. `barrier_sensitivity_envelope.csv` brackets these fields across the retained center and nearby conditional profile vectors under the declared working-cost threshold; it is not a confidence band. The code's G00/H0 fields are reference free energies, not physical activation enthalpies.","",
        "## Unsupported channels and model limits", "",
        "- The pinned native pair has no independent final unstable-cleavage classification. The added geometric ligament-failure endpoint is a physical model check, but it cannot be assumed to equal Ast's abrupt cleavage event. Table 1 LEFM/basic-J cleavage means and no-cleavage endpoints therefore remain validation targets rather than fitted residuals; unknown subgroup counts preclude a fabricated binomial likelihood. Separate endpoint-validation trajectories extend the independent rate-controlled loading to the observed endpoint's initial-elastic equivalent displacement; the observed endpoint does not prescribe a crack event or enter the fit.",
        "- Relative stiffness uses a frozen-state elastic compliance probe because the experimental oscillation frequency was not recovered. Fig. 2a displacement origins are bounded relative-shift nuisances. The A2 temperature is described only as room temperature; its 21 °C model overlay is explicitly representative.",
        "- The spatial model uses signed forward-minus-reverse Peierls/Taylor rates and transport, whereas the reduction uses forward rates and mean-field translation. Parameter transfer is therefore assessed with paired histories and not assumed from the common EXP-floor surfaces.","",
        "## Reproduction and runtime", "",
        f"- Parameter table: `{parameter_file.name}`; actual stage call counts and accumulated worker runtime: `runtime_accounting.json`.",
        f"- Recorded search/refinement forward cases: {runtime['total_completed_search_refine_case_calls']}; recorded paired confirmation forward cases: {runtime['total_confirmation_case_calls']}. Timings in the JSON are accumulated across workers and are not campaign elapsed time.",
        f"- Execution environment: Python `{environment['python'].split()[0]}`, NumPy `{environment['numpy']}`, SciPy `{environment['scipy']}`, Matplotlib `{environment['matplotlib']}` on `{environment['machine']}`; full receipt in `environment_receipt.json`.",
        "- Machine-readable condition/family costs and paired toughness comparisons: `residual_contributions.csv` and `toughness_predictions.csv`.",
        "- `source_scenario_comparison.csv` gives condition-level onset shifts under the two source readings for both model levels.",
        "- Use the pinned native checkout and the supplied archive. The commands below invoke no 2-D path:","",
        "```sh",
        "python -m experimental_fits.ast2018.run_all --native-source /path/to/pinned/Arrhenius_FEM_CZM_MPZ --archive Ast2018_tungsten_joint_fit_data.zip --run-dir runs/ast2018_joint_fit_v1 --workers 8 --primary-starts 128 --alternative-starts 32",
        "```", "",
        "`run_all.py` is the resumable complete-campaign launcher. `search_bounded.py`, `refine.py`, and `confirm.py` remain available as separate resume/analysis commands. All physical interpretation above is supported by the saved direct trajectories and event records, not by an unfinished optimizer call.",""])
    report=run_dir/"FIT_REPORT.md"
    report.write_text("\n".join(lines))
    return report
