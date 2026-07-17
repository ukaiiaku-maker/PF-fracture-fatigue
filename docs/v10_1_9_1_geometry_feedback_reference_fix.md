# v10.1.9.1 geometry-feedback reference correction

v10.1.9 stored the effective radius at the first crack advance as the source-feedback reference. In the high-temperature DBTT case, the largest early emitted population already existed at first passage, so subsequent radii did not exceed that reference and every nonzero geometry gain remained inactive.

v10.1.9.1 preserves the same first-passage calculation but separates **arming time** from **geometry reference**:

- feedback is disabled through the first fired crack-advance record;
- the fixed reference radius is the original unblunted `r0`;
- the first-advance radius is retained only as a diagnostic;
- on the next kinetic interval, the already-developed blunting is evaluated relative to `r0`;
- source-capacity exposure follows the irreversible running maximum of normalized blunting;
- the analyzer exits with an inactive-parameter error if no nonzero gain increases the high-temperature capacity ratio.

No Arrhenius barrier, source-refresh length, back-stress scale, shielding cap, cleavage law, or wake-shielding setting is changed.
