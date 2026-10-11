# Current-PF full-order FEM/CZM companion: one-step result

## Scope and provenance

This is a bounded compatibility calculation, not a reconstruction of the
unavailable historical `e239710...` input. The separate companion branch
`codex/current-pf-companion` at `fa24f7b` calls the native full-order
theta-zero adaptive-CZM builder using the immutable current PF family with
SHA-256 `d41b08f69a...` and mechanical fingerprint `1447653d19...`. The
historical checkout and its `a85b57...` guard remain unchanged.

The independent backup gate passed before execution. Source, dependencies,
prior evidence, and historical reference output were copied to `/dev/disk3s5`
and SHA-256 verified. Nothing was deleted.

## Supported result

The solver completed one accepted step at 1000 K:

| quantity | value |
|---|---:|
| mesh | 2,020 nodes, graded tip size 1.57937 µm |
| prescribed opening | 0.2 µm |
| time increment | 8.4 s |
| reaction | 2650.777674 N |
| native J | 0.0881706431 J/m² |
| native K_J | 0.198053581 MPa√m |
| tip stress | 0.0789932041 GPa |
| backstress | 0.0694658601 GPa |
| cleavage progress B | 8.42047e-101 |
| initial/final crack tip | 0.500/0.500 mm |
| cohesive advances | 0 |

This is a finite no-event endpoint at a low first load step. It qualifies the
current-input full-order execution path and its saved state; it does not imply
a developed-load fracture threshold and no event was inserted or forced.

## Saved mechanical state

Raw output includes the accepted step table, native J/K and traction-probe
diagnostics, crack/front coordinates, cohesive elements and empty advance log,
MPZ snapshots, energy/emission ledgers, and a 3,900-point bulk state. The bulk
arrays contain explicit mobile and retained densities; the constitutive state
was updated twice. This model performs local full-field Peierls/Taylor state
evolution without spatial bulk-density transport.

The authoritative run argument, parity contract, normalized summary, and
metadata-normalization record all identify theta = 0°. Inherited lower-level
diagnostic summaries still carry a legacy 45° label; the normalization record
marks these as non-authoritative metadata. No FEM, cohesive, hazard, mesh, or
event state was changed by normalization.

## Integrity

The complete 2.8 MiB raw directory contains 39 files. It was copied after the
run to the independent backup and every file passed SHA-256 verification.
Selected raw hashes are recorded in the adjacent JSON. The raw output remains
outside Git; this compact report is the reviewable evidence layer.
