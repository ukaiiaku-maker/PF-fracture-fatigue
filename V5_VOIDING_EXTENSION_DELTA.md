# V5 voiding extension delta

The V5 contribution is limited to explicit void state and kinetics, cavity free-surface awareness, cavity remesh and connection transactions, and downstream void-front birth/continuation. It consumes the qualified V13 front, multi-tip ownership, hazard renewal, energy gate, RNG, and checkpoint core.

The extension adds no fracture criterion, scientific tolerance, material-specific void fit, or alternative front-engine constructor. The common reference void-kinetics row stays separate from the four immutable fracture rows.

Paper deltas are defined as:

- branching increment: `O[MULTI_TIP] - O[SINGLE_TIP]`
- voiding increment: `O[MULTI_TIP_PLUS_VOIDING] - O[MULTI_TIP]`
- combined increment: `O[MULTI_TIP_PLUS_VOIDING] - O[SINGLE_TIP]`

The tip-radius law remains `r_tip = r0 + c_blunt*b*local_weighted_accumulated_slip`; cavity radius never supplies `r_tip`.
