# Analytical thermodynamic joint-search decision

**Status:** complete analytical inverse-design screen; no physical simulation was launched.

The successful P25 and P40 300 K fatigue responses can be preserved while restoring the zero-stress 300 K cleavage barrier to 1–2 eV. A standard single EXP-floor cannot do both under the frozen local-slope gate; the low-stress guard is required.

The retained set contains 12 guard-plus-linear-entropy candidates. Intrinsic F0 contains ceramic-like and weak-T responses, plus accessible unclassified controls. Conditional F1 closes only at a limited subset of conditions and supports no DBTT-like or Peak-T claim. F2 remains unavailable.

The preferred 30–40 kB emission regime was searched with a secondary entropy basis and bounded ±15 kB activation heat capacity. Zero candidates retained resolved emission competition through 1200 K; DeltaCp is therefore tested but is not used by the retained intrinsic candidates. This is a negative bounded result, not evidence that every possible heat-capacity surface fails.

## Retained candidates

- `P25_TJBS_S_014188` — ACCESSIBLE_UNCLASSIFIED; F1: STATE_UNRESOLVED_ANALYTICAL_CANDIDATE
- `P25_TJBS_S_028493` — ACCESSIBLE_UNCLASSIFIED; F1: STATE_UNRESOLVED_ANALYTICAL_CANDIDATE
- `P25_TJBS_S_039907` — ANALYTICAL_JOINT_CANDIDATE_CERAMIC_LIKE; F1: STATE_UNRESOLVED_ANALYTICAL_CANDIDATE
- `P25_TJBS_S_047409` — ANALYTICAL_JOINT_CANDIDATE_CERAMIC_LIKE; F1: STATE_UNRESOLVED_ANALYTICAL_CANDIDATE
- `P25_TJBS_S_016645` — ANALYTICAL_JOINT_CANDIDATE_WEAK_T; F1: STATE_UNRESOLVED_ANALYTICAL_CANDIDATE
- `P25_TJBS_S_064760` — ANALYTICAL_JOINT_CANDIDATE_WEAK_T; F1: STATE_UNRESOLVED_ANALYTICAL_CANDIDATE
- `P40_TJBS_S_037582` — ACCESSIBLE_UNCLASSIFIED; F1: STATE_UNRESOLVED_ANALYTICAL_CANDIDATE
- `P40_TJBS_S_040211` — ACCESSIBLE_UNCLASSIFIED; F1: STATE_UNRESOLVED_ANALYTICAL_CANDIDATE
- `P40_TJBS_S_008683` — ANALYTICAL_JOINT_CANDIDATE_CERAMIC_LIKE; F1: STATE_UNRESOLVED_ANALYTICAL_CANDIDATE
- `P40_TJBS_S_049735` — ANALYTICAL_JOINT_CANDIDATE_CERAMIC_LIKE; F1: STATE_UNRESOLVED_ANALYTICAL_CANDIDATE
- `P40_TJBS_S_053775` — ANALYTICAL_JOINT_CANDIDATE_WEAK_T; F1: STATE_UNRESOLVED_ANALYTICAL_CANDIDATE
- `P40_TJBS_S_064772` — ANALYTICAL_JOINT_CANDIDATE_WEAK_T; F1: STATE_UNRESOLVED_ANALYTICAL_CANDIDATE

Complete parameters and numerical ranges are in `retained_candidate_parameter_rows.csv`, `retained_joint_candidates.csv`, and `analytical_joint_search_decision.json`. All labels are prospective model-response classifications, not validated material archetypes.
