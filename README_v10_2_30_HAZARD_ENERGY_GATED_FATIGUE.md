# v10.2.30 hazard-energy-gated persistent-site fatigue

This point release extends the audited v10.2.29 anisotropic PF/sharp-front fatigue stack. It does not use the incomplete FEM/CZM fatigue implementation.

## Preserved production model

- exact four canonical persistent-site parameter rows: peak, DBTT, weak-T, and ceramic-like;
- anisotropic cubic mechanics and cleavage-direction competition;
- one active nonbranching front;
- signed retained shielding, zero mobile shielding, and no wake shielding;
- persistent emission sites with no finite inventory, depletion, refresh, or explicit recovery;
- state-coupled cyclic first-passage integration and consumed-cycle accounting;
- exponential stochastic cleavage threshold and threshold-scaled mean-preserving event-length proposal.

## Event-length correction

Cleavage first passage remains the only stochastic trigger. The continuum K-squared-over-E-prime comparison is diagnostic only and cannot suppress or rescale the cleavage hazard.

After first passage, the existing stochastic event-length draw is treated as an upper proposal. The committed distance is the largest mesh-resolved sharp-wake extension satisfying the fixed-opening elastic-energy balance at the event Kmax. The probe-field release is scaled by `(K_event/K_probe)^2` under the existing fixed-local-DeltaK architecture.

The trial event resistance is derived only from active production hazard quantities:

`Gamma_hazard = gamma_relative * m_hits * DeltaG_cleave_effective / b^2`

This is an explicit, parameter-free mapping from the active hazard free-energy surface to event work per new crack area. It is part of the v10.2.30 physical qualification and is not treated as previously validated surface-energy data. No `Gc0_athermal`, generic fracture-resistance configuration, toughness floor, Paris law, or fitted fracture-energy coefficient is used.

During waiting cycles the geometric tip is stationary while the persistent-site mobile/retained state evolves. Once an event is admitted, the sharp-wake geometry and moving-frame MPZ translate atomically by the same committed distance. A first-passage attempt with no mesh-resolved admissible increment is consumed as a nonpropagating attempt; its stochastic threshold is not restored.

## Qualification gate

The low-level one-class primitive is:

`scripts/run_v10_2_30_three_deltaK_energy_gate_qualification.sh`

It requires one matching monotonic first-passage reference and supports class-specific DeltaK fractions for bracketing and numerical refinement.

The release gate for the canonical material set is:

`scripts/run_v10_2_30_four_class_three_deltaK_energy_gate_qualification.sh`

It requires `PEAK_REFERENCE_ROOT`, `DBTT_REFERENCE_ROOT`, `WEAKT_REFERENCE_ROOT`, and `CERAMIC_REFERENCE_ROOT`. The wrapper runs the exact peak, DBTT, weak-T, and ceramic-like rows with separate class seed namespaces while retaining common random numbers between each class's primary and refined event-length trials.

The four-class gate requires:

- exact fixed-DeltaK control;
- at least one propagated and one censored case for every material class;
- no independent fracture energy;
- no change to first-passage kinetics;
- no non-energy geometry veto;
- mesh-resolved topology for every committed event;
- event energy closure;
- common-seed event-length convergence under 0.10 to 0.05 trial-fraction refinement for every class;
- at least one positive committed event truncated by the energy balance across the four-class matrix;
- unique class seed namespaces;
- projected `da_x/dN` and path-length `ds/dN` outputs;
- anisotropic direction and relative-orientation audits.

The final machine-readable release gate is `v10_2_30_four_class_qualification_gate.json`. The four-class production `da/dN` sweep remains blocked unless its `pass` field is true.

The dedicated v10.2.30 workflow compiles the additive point release and runs the corrected energy-gate, transactional-engine, fixed-DeltaK, one-class and four-class qualification-tool, and inherited v10.2.29 cyclic-accounting contracts. The repository-wide legacy preflight is tracked separately.
