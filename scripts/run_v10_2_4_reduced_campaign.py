#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import traceback
from typing import Any

from arrhenius_fracture.reduced_campaign_v1024 import (
    MODEL_ID,
    evaluate_candidate,
    generate_candidate_rows,
    write_manifest_row,
)
from arrhenius_fracture.reduced_shared_state_v1023 import SharedReducedConfig


def _worker(payload: tuple[dict[str, Any], str, dict[str, Any]]):
    row, atlas, cfg_dict = payload
    try:
        score, _runs = evaluate_candidate(
            row,
            atlas,
            SharedReducedConfig(**cfg_dict),
        )
        return {"ok": True, "row": score}
    except Exception as exc:
        return {
            "ok": False,
            "candidate_id": row.get("candidate_id"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "row": row,
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--promote", type=int, default=16)
    parser.add_argument("--Kdot", type=float, default=0.005)
    parser.add_argument("--Kmax", type=float, default=80.0)
    parser.add_argument("--dK", type=float, default=0.05)
    parser.add_argument("--target-extension-um", type=float, default=5.0)
    parser.add_argument("--checkpoint-da-um", type=float, default=5.0)
    parser.add_argument("--transport-mode", default="validated_scalar")
    args = parser.parse_args()

    if not args.atlas.is_file():
        raise SystemExit(f"atlas is missing: {args.atlas}")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "promoted_manifests").mkdir()

    cfg = SharedReducedConfig(
        Kdot_MPa_sqrt_m_s=float(args.Kdot),
        Kmax_MPa_sqrt_m=float(args.Kmax),
        max_dK_step_MPa_sqrt_m=float(args.dK),
        target_extension_um=float(args.target_extension_um),
        checkpoint_da_um=float(args.checkpoint_da_um),
        transport_mode=str(args.transport_mode),
    ).validate()
    candidates = generate_candidate_rows(int(args.samples), int(args.seed))
    _write_csv(args.out / "generated_candidates.csv", candidates)
    (args.out / "campaign_config.json").write_text(
        json.dumps(
            {
                "schema": MODEL_ID,
                "atlas": str(args.atlas.resolve()),
                "samples": len(candidates),
                "seed": int(args.seed),
                "workers": max(int(args.workers), 1),
                "promote": int(args.promote),
                "shared_state_config": asdict(cfg),
                "candidate_acceptance_requires_2d_validation": True,
                "constitutive_K_shield_cap_applied": False,
            },
            indent=2,
        )
    )

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    progress_path = args.out / "campaign_progress.jsonl"
    payloads = [(row, str(args.atlas), asdict(cfg)) for row in candidates]

    with progress_path.open("w") as progress:
        if max(int(args.workers), 1) == 1:
            iterator = enumerate(map(_worker, payloads), start=1)
            for count, result in iterator:
                progress.write(json.dumps(result) + "\n")
                progress.flush()
                if result["ok"]:
                    completed.append(result["row"])
                else:
                    failures.append(result)
                print(
                    f"[{count}/{len(payloads)}] {result.get('candidate_id', result.get('row', {}).get('candidate_id'))} "
                    f"ok={int(result['ok'])}",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=max(int(args.workers), 1)) as pool:
                futures = {pool.submit(_worker, payload): payload[0]["candidate_id"] for payload in payloads}
                for count, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    progress.write(json.dumps(result) + "\n")
                    progress.flush()
                    if result["ok"]:
                        completed.append(result["row"])
                    else:
                        failures.append(result)
                    print(
                        f"[{count}/{len(payloads)}] {futures[future]} ok={int(result['ok'])}",
                        flush=True,
                    )

    completed.sort(
        key=lambda row: (
            not bool(row.get("strict_reduced_pass", False)),
            float(row.get("objective", math.inf)),
        )
    )
    _write_csv(args.out / "candidate_scores.csv", completed)
    (args.out / "candidate_failures.json").write_text(json.dumps(failures, indent=2))

    promoted = completed[: min(max(int(args.promote), 0), len(completed))]
    _write_csv(args.out / "promoted_candidates.csv", promoted)
    for rank, row in enumerate(promoted, start=1):
        write_manifest_row(
            args.out / "promoted_manifests" / f"rank_{rank:03d}_{row['candidate_id']}.csv",
            row,
        )

    strict_count = sum(bool(row.get("strict_reduced_pass", False)) for row in completed)
    assessment = {
        "schema": MODEL_ID,
        "status": "complete",
        "samples_requested": len(candidates),
        "samples_completed": len(completed),
        "samples_failed": len(failures),
        "strict_reduced_pass_count": strict_count,
        "promoted_count": len(promoted),
        "best_candidate_id": promoted[0]["candidate_id"] if promoted else None,
        "best_objective": float(promoted[0]["objective"]) if promoted else None,
        "reduced_result_is_final_parameterization": False,
        "next_required_gate": "cap-free 2-D endpoint and mechanism ablations",
    }
    (args.out / "campaign_assessment.json").write_text(json.dumps(assessment, indent=2))
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
