# Corrected-transfer native signed-bulk refit

The native material binding now transfers the fitted blunting length instead of
using the historical hard-coded 5.5577 um value. The corrected parent uses
17.3533 um. All matrices below contain 16 developed three-event trajectories
with element bulk plasticity and signed crack-swept GND absorption enabled.

| Native matrix | log10 RMSE vs experiment | log10 RMSE vs 1-D | R-block bias span |
|---|---:|---:|---:|
| old incomplete transfer | 0.6327 | 0.4246 | 1.3415 |
| corrected 17.35 um transfer | 0.5121 | 0.2540 | 1.2485 |
| corrected transfer plus state-informed opening refit | **0.3836** | **0.1663** | **0.9333** |

The final native block biases are -0.435, -0.269, +0.041, and +0.498 decades
for R=0.1, 0.3, 0.5, and 0.7. The corresponding slopes are 2.474, 2.068,
2.341, and 2.429. The refit therefore improves vertical agreement and R collapse,
but produces slopes below the measured approximately 2.9--3.8 range.

The opening refit varied only the authorized smooth sloping-floor coordinates on
developed native stress histories, then underwent a fresh 16-case native replay.
Its amplitude and exponent reached the declared acquisition bounds. It is a
useful constrained result, not evidence that further opening-only fitting will
necessarily recover the experimental slopes. The remaining discrepancy points
to the evolution/transfer of plastic shielding and blunting, rather than a simple
constant opening-rate offset.

