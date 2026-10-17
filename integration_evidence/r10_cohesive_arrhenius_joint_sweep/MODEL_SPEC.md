# Additive cohesive–Arrhenius R10 family

## Identity and scope

This directory belongs to `D_COHESIVE_ARRHENIUS_EXP_PLASTICITY`. It is additive to, and does not rewrite, the retained EXP_floor bank. The reconciled base is full commit `97a1e48fa556f784b2fb5203916da2d17af98732` on the dedicated branch `codex/r10-cohesive-arrhenius-joint-sweep`.

The supported execution identities are:

- `EXP_FLOOR_REFERENCE`: the unmodified retained control.
- `COHESIVE_BARRIER_ONLY`: the new elementary opening barrier consumed by the retained R10 prescribed-K mechanics, cooperative `gammainc(q, k tau)/tau` completion, marked event/reward law, and persistent/reversible dislocation state.
- `COHESIVE_KINETIC_CZM`: the same elementary barrier plus the actual nonlinear interface traction, consistent tangent, irreversible accepted-event rupture state, unloading/reloading, and non-healing contact.
- `COHESIVE_RATE_INDEPENDENT_CONTROL`: reserved comparison identity. It is not an Arrhenius result and is not exercised by the current evidence.

Unsupported or missing identities fail closed. The final bound identity appears in native runtime receipts. Old cohesive/EXP_floor fields retained in a parent row are inactive for opening in this family.

## Constitutive definition

The Mode-I free energy, traction, and tangent are

`psi = Gamma(T) [1 - (1+x) exp(-x)]`,

`t = Gamma(T)/delta0 x exp(-x)`, and

`dt/ddelta = Gamma(T)/delta0^2 (1-x) exp(-x)`,

where `x=delta/delta0` and `delta0=Gamma_ref/(e t_hat_ref)`. `delta0` is therefore derived, not independently sampled. The two declared thermal subfamilies are constant `Gamma` and `Gamma(T)=Gamma_ref-(T-300 K) S_Gamma` at fixed `delta0`. Raw positivity is checked at all seven temperatures.

For `0 < sigma/t_hat < 1`, the traction-controlled minimum and saddle use the real `W_0` and `W_-1` Lambert branches. The activation barrier is their direct tilted-potential difference, with activation volume `A_star delta0 (x_s-x_m)`. Exact zero traction reports the formal `A_star Gamma` barrier and infinite saddle/activation volume, while its mechanically inadmissible opening rate is zero. Compression is `INADMISSIBLE_COMPRESSION`; loss of metastability is `BARRIERLESS_SPINODAL`. The near-spinodal `3/2` asymptotic avoids cancellation. No positive barrier floor is added.

The finite-compliance reference solves the stationary points of `psi + k_env(Delta-delta)^2/2`. A convex combined landscape is `NO_ESCAPE_SADDLE`, not a zero barrier. `k_env` is a mechanics condensation input and is not searched.

The triangular-bilinear and Morse barriers are analytical shape controls matched in `Gamma`, `t_hat`, `A_star`, and prefactor. They are not separate search campaigns.

## Parameter and site bookkeeping

The prospective priors are frozen in `prospective_search_protocol.json`. `Gamma_ref`, `t_hat_ref`, and the zero-traction activation energy determine `delta0` and `A_star`; activation area is never a numerical element area. Attempt frequency is fixed at the inherited `1e12 s^-1`. Site multiplicity and the cooperative completion operator remain separate owners, so neither is multiplied into the elementary barrier.

Every offspring retains one complete R10 parent dislocation vector. Emission, Peierls, Taylor, source geometry, activation entropies (including negative values), moving state, and reward laws are unchanged. Parent opening EXP_floor fields are provenance only.

## Mechanics ownership

| Contribution | Owner in barrier-only | Owner in kinetic CZM |
|---|---|---|
| Cohesive traction and surface storage | Not resolved; retained prescribed-K operator | Nonlinear interface assembly |
| Elastic redistribution | Retained prescribed-K operator | Bulk FEM plus interface tangent |
| Signed source backstress/transport | Persistent/reversible MPZ | Same persistent/reversible MPZ |
| Accepted rupture event | Inherited cooperative marked event | Same event calls `accept_rupture` |
| Contact | Not applicable | Separate numerical compression penalty |
| Surface-energy charge | Barrier only; no extra event bill | Interface potential only; no second Griffith cost |

The barrier-only legacy 30 GPa numerical strength contract remains independent of `t_hat`. The kinetic interface does not use a deterministic peak-traction failure rule; crossing the peak changes tangent but does not itself rupture the interface.

## Reproduction

Use the Python 3.12 environment recorded in `base_reconciliation.json`:

```bash
python scripts/run_r10_cohesive_campaign.py qualification
python scripts/run_r10_cohesive_campaign.py acquisition
python scripts/run_r10_cohesive_campaign.py expansion
python scripts/run_r10_cohesive_campaign.py spatial --candidate-id R10D_COH_00167
python scripts/run_r10_cohesive_campaign.py native --candidate-id R10D_COH_00167
python scripts/run_r10_cohesive_campaign.py ledger
```

All stages are additive and idempotent. A 32,768-row analytical allocation, a 4,096-row state-aware allocation, and a 48-row native queue are explicitly not execution counts. Native outputs and restart state live under `runs/r10_cohesive_arrhenius_joint_sweep/`.

## Evidence limits

Analytical class/mechanism strata are acquisition labels, not causal classifications. A 2-D qualification proves consumption of the nonlinear interface and its transactions; it is not a nine-cell scientific result. Only trajectories reaching the 20–60 micrometre event window are developed-rate evidence. Unfilled portfolio cells remain `NOT_YET_FOUND`.

The post-allocation all-secant audit found zero rows whose five adjacent analytical secants all lie in 3.4–4.6 over at least two rate decades. Near-four median or endpoint slopes are therefore not promoted. The current native ladder tests whether retained state evolution changes that negative frozen-opening result; it is not pre-labelled a success.
