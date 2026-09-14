# Entropy sign and units audit

The v2 convention is `Delta S^dagger/k_B = -(partial_T Delta G^dagger)/k_B`. Thus `Delta S_emit^dagger/k_B=-40` means a barrier derivative of `+40 k_B` at fixed stress. The v1 code used positive activation entropy for its preferred bank, while its prose was sign-ambiguous. V2 stores both quantities in every thermodynamic table and uses `DIRECT_FREE_ENERGY_SURFACE_NO_PREFACTOR_DOUBLE_COUNTING`.
