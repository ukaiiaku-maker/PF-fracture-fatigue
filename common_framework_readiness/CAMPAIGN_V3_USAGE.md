# Disabled prospective campaign v3

The v3 campaign is an executable construction/audit plan, not production authorization. Every record retains `production_execution_authorized=false`; target machine and output root are intentionally unset.

Build and audit the complete disabled plan:

```console
python scripts/build_prospective_campaign_v3.py --disabled --audit --dry-run
```

Select one or more blocks without launching them:

```console
python scripts/build_prospective_campaign_v3.py --disabled --audit --dry-run --block P8 --block P9
```

Audit only cases not listed as completed in a resume-state JSON:

```console
python scripts/build_prospective_campaign_v3.py --disabled --audit --dry-run --resume-state path/to/resume.json
```

The resume file has the minimal form `{"completed_case_ids": ["case-id"]}`. Output validation requirements and peak/retained storage estimates are stored per case. Resource admission must be recomputed on an authorized target machine as peak transient space plus reserve; summed retained-output estimates are not a launch preflight.

The 48-case P9 block compares original activated opening, temperature-matched activated bilinear, temperature-matched autonomous irreversible bilinear, and reference-temperature-frozen autonomous bilinear controls. Fitted bilinear cases deliberately remain blocked until activation area and mechanical cohesive support are physically assigned; no specimen stiffness, work, separation, or kinetic scale is fabricated.
