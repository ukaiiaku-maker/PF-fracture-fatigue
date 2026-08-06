# V11 Analytical Directional Competition Validation

## Scope

These tests validate the disconnected analytical competition model only. They do
not create cracks, mutate FEM geometry, resolve kernels, write production
checkpoints, or couple competition into the production 2-D driver.

## Contract coverage

- A single candidate and a parameter matrix of initial actions, rates, and time
  intervals are checked against the exact constant-rate first-passage solution.
- Nonpositive signed directional J is checked to produce zero positive J, zero
  directional K, and zero topology-driving hazard rate.
- Production hazard previews are repeatable and leave the supplied engine bytewise
  unchanged.
- Candidate identity is derived from canonical crystallographic and normalized
  global-direction data, including canonical signed zero.
- Shuffled candidate inventories produce identical serialized state, rates,
  completion times, proposals, and tie outcomes.
- Temporal priority is independent of the seed. The seed is exercised only for
  alternatives whose completion times are numerically degenerate.
- Reservation release consumes nothing; acceptance consumes exactly the event IDs
  declared by each one-arm or two-arm proposal; overlapping reservations fail.
- Canonical serialization is checked for round-trip equality, deterministic bytes,
  malformed state rejection, active/released reservations, consumed events, and
  legacy one-tip compatibility.
- The committed branching-disabled reference and canonical legacy one-tip state
  hashes are checked without regenerating either authoritative artifact.

## Commands

The milestone validation commands are:

```bash
conda run -n arrhenius-sharp-front-v10 python -m pytest -q \
  tests/test_crack_network_v11.py \
  tests/test_directional_competition_v11.py \
  tests/test_directional_competition_transactions_v11.py \
  tests/test_directional_competition_serialization_v11.py \
  tests/test_directional_competition_symmetry_v11.py
```

The unchanged production-contract selection from the preceding milestone is run
separately, followed by the complete test suite. Exact results are recorded in the
milestone handoff rather than embedded here, so this document does not become
stale when unrelated tests are added.

## Preserved reference hashes

- Branching-disabled authoritative output:
  `fbd35339b09b685e7a524447b4e6414b1b3364c3cb7f7c012b478365f02af191`
- Canonical legacy one-tip compatibility state:
  `1c5025e451d4c667ac51288e45d00c9e27b42411c35874eab3ff9b13585bed5a`

Generated outputs used by tests are confined to fresh temporary directories.
