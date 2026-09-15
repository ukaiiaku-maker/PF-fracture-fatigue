# Named response parameterizations, catalog V2.0.0

`LEGACY` identifies the established four-class parameter set used by the completed canonical PF campaign. `CORRECTED_JOINT_THERMODYNAMIC_V2` identifies the corrected negative-emission-entropy, coupled opening/emission/transport search generation. The displayed V2 designation is a scientific parameterization generation and is independent of historical internal filenames such as `oneD_v2_focused_*`.

The unsuffixed aliases `Peak`, `DBTT`, `weakT`, and `ceramic` retain their established resolution. V2 is a prospective successor generation under evaluation. It does not supersede Legacy and is available only through explicit aliases. No `Peak_V2` alias is assigned because no Peak-T candidate passed F1B or F2R in the corrected frozen search domain. No preferred `weakT_V2` or `ceramic_V2` alias is assigned while their P25 and P40 alternates remain under evaluation.

This catalog changes naming, lookup, and provenance only. It changes no physics, runs no simulation, and leaves the canonical source registry byte-identical.

## Legacy versus V2

| Family | Legacy alias | V2 aliases | Selection policy |
|---|---|---|---|
| DBTT | `DBTT` | `DBTT_V2 / DBTT_V2_P40` | Explicit opt-in; preferred unsuffixed V2 alias is assigned only for DBTT |
| weak-T | `weakT` | `weakT_V2_P25 / weakT_V2_P40` | Explicit opt-in; preferred unsuffixed V2 alias is assigned only for DBTT |
| ceramic-like | `ceramic` | `ceramic_V2_P25 / ceramic_V2_P40` | Explicit opt-in; preferred unsuffixed V2 alias is assigned only for DBTT |

## Evidence cards

- [Peak](parameterization_evidence_cards/Peak.md)
- [DBTT](parameterization_evidence_cards/DBTT.md)
- [weak-T](parameterization_evidence_cards/weakT.md)
- [ceramic-like](parameterization_evidence_cards/ceramic.md)
- [DBTT V2](parameterization_evidence_cards/DBTT_V2.md)
- [DBTT V2-P40](parameterization_evidence_cards/DBTT_V2_P40.md)
- [weak-T V2-P25](parameterization_evidence_cards/weakT_V2_P25.md)
- [weak-T V2-P40](parameterization_evidence_cards/weakT_V2_P40.md)
- [ceramic-like V2-P25](parameterization_evidence_cards/ceramic_V2_P25.md)
- [ceramic-like V2-P40](parameterization_evidence_cards/ceramic_V2_P40.md)

The JSON registry is authoritative for naming metadata and embeds exact source-row strings. At load time, every embedded row and hash is checked against its sealed source registry.
