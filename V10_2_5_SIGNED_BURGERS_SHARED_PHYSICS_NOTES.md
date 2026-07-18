# v10.2.5 signed-Burgers shared physics

## Status

The v10.2.4 parameter campaign is **stopped**. Its launch scripts now fail
closed. The old search must not be resumed or used for promotion.

v10.2.5 implements the shared software architecture required for a physically
signed population and shielding interaction. It does **not** include a fabricated
or fitted shielding kernel. Production fracture/fatigue runs remain blocked until
real 2-D unit-response and source-normalization artifacts are generated and
validated.

## Why v10.2.4 is invalid

The cap-free A0002333 calculation exposed two separate defects:

1. Replay did not reconstruct the complete production configuration. In
   particular, the 30 GPa local opening-strength limit was replaced by
   `sigma_cap=0`, and several front, MPZ, kinetic, transport, campaign, and
   anisotropic settings were omitted.
2. A source-site activation was transferred directly into one unsigned coherent
   dislocation-line unit. The shielding operator then used the analytic
   `Gb/sqrt(2*pi*x)` form with channel factors `(1,1)`. Thousands of hazard
   opportunities therefore became thousands of coherently shielding lines.

The removed K-shield clip had hidden the second defect. Reintroducing a clip or
adding a fitted attenuation factor is not permitted.

## Shared state model

`arrhenius_fracture.signed_burgers_shared_v1025` defines one physical state law
used by both loading paths:

- `mobile_positive`, `mobile_negative`;
- `retained_positive`, `retained_negative`;
- positive/negative accumulated-slip fields;
- positive/negative wake fields.

The existing unsigned totals are maintained as sums of the two species. They are
used for forest density, Taylor back stress, source exhaustion, transport,
recovery, and blunting. Shielding uses the signed difference.

For system `s` and bin `i`,

```
N_signed[s,i] = N_positive[s,i] - N_negative[s,i]
K_shield = sum(H_active[s,i] * N_signed_active[s,i])
         + sum(H_wake[s,i]   * N_signed_wake[s,i])
```

A negative kernel coefficient or a reversed Burgers sign naturally produces
antishielding. There is no constitutive K-shield cap.

## Emission normalization

`source_sites_per_system` remains a count of statistical nucleation
opportunities. It is not automatically interpreted as coherent dislocation-line
content.

A kernel artifact must provide a mechanically derived
`activation_to_line_content_by_system` conversion. Accepted derivations are:

- a 2-D unit-slip to line-content calculation;
- process-zone geometry and attainable line spacing;
- front thickness and source geometry.

The artifact must also provide physical source-capacity bounds. A material
manifest outside those bounds is rejected before evolution. This deliberately
rejects the old factor-three search around approximately 4,600--9,300 sites when
the derived geometry supports only a much smaller line inventory.

## 2-D signed shielding kernel

The required artifact schema is
`v10.2.5_2d_unit_signed_shielding_kernel`.

For every active/wake system and spatial bin, the 2-D calculation must apply both
positive and negative unit signed perturbations and record

```
H[s,i] = delta_K_tip / delta_signed_line_content.
```

`scripts/build_v10_2_5_signed_shielding_kernel.py` requires both signs and checks
that their normalized responses are antisymmetric/linear within a specified
tolerance. It rejects incomplete matrices and normalization files that are not
identified as mechanically derived.

No default `(1,1)` projection and no placeholder kernel are provided.

## Exact replay parity

`arrhenius_fracture.reduced_shared_state_v1025.ExactProductionConfig` reconstructs
all serialized production settings:

- complete front configuration: `r0`, `L_pz`, `da`, `sigma_cap`, `m_hits`,
  `tau_c`, plus any other serialized public fields;
- every `MPZConfig` field;
- every `KineticTipConfig` field;
- every `AnisotropicEmissionConfig` field;
- transport mode;
- campaign backstress and refresh scales;
- `G`, `nu`, and `b`;
- the signed shielding kernel.

The reconstructed engine is compared field-by-field with the serialized
configuration and aborts on any mismatch. The 30 GPa `sigma_cap` is preserved as
a local cohesive/strength limit. It is explicitly distinct from the removed
K-shield cap.

`state_equivalence_trace_v1025` additionally records signed resolved shear for
each channel and all positive/negative state arrays. The replay cannot infer or
replace missing signs.

## Common monotonic and fatigue entry point

Both loading paths use

```
python -m arrhenius_fracture.sharp_front_v10_2_5
```

The only loading-path difference is whether `--fatigue-cycles` is present. Both
paths install the same `SignedBurgersAnisotropicTipEngine`, signed population,
transport law, moving-frame update, source normalization, and 2-D shielding
kernel.

The entry point requires

```
--signed-shielding-kernel PATH
```

or

```
SIGNED_SHIELDING_KERNEL_JSON=PATH
```

and exits if the artifact is absent or invalid.

## Required validation sequence

1. Generate candidate-independent positive/negative 2-D unit perturbation
   responses on the production active and wake grids.
2. Derive activation-to-line content and physical source-capacity bounds from
   mechanics/geometry, not parameter fitting.
3. Build and inspect the signed kernel artifact.
4. Run a signed 2-D case with `SIGNED_STATE_TRACE=1`.
5. Pass exact replay configuration and final-array equivalence.
6. Run monotonic temperature-fracture and cyclic-fatigue smoke cases through the
   same entry point and verify signed shielding/antishielding diagnostics.
7. Only then define a new physically bounded parameter-search region.

A0002333 may be retained as a legacy capped-response reference, but it is not an
uncapped promotion anchor.
