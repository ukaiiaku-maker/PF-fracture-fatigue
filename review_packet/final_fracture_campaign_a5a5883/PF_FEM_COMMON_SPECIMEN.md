# PF–FEM common-specimen audit

The retained Peak/1000 K runs are not a backend-only comparison. Their body,
opening history, cubic elasticity, mesh controls, orientation and nominal seed
match, but their accommodation states and observation supports do not.

At the first common opening (0.2 µm), PF reports
43567.9718 N/m and FEM/CZM reports
2661.22241 N/m (ratio 16.3714).
The executed PF arm is `tip_only` with
Lpz=1 µm; the FEM arm is
`full_field` with
Lpz=50 µm. The saved parity
metadata incorrectly calls the PF arm full-field; the numerical state itself
was not altered.

The accommodation-disabled elastic reference passed=True,
with maximum relative PF–FEM difference
0.000e+00. This closes the
geometry, symmetric-opening, force-unit and base elastic-operator questions for
the shared diffuse-notch specimen. It does not establish first-passage or
post-event parity.

Next, the histories must be compared under identical tip-only accommodation,
then under the existing full-field Peierls–Taylor option with the same process
zone and J supports. Any event-level cohesive claim still requires a native
FEM cohesive event and post-event equilibrium.
