# Fatigue K-coordinate decision

`COMMON_FATIGUE_AND_MONOTONIC_K_COORDINATE_QUALIFIED` applies: monotonic and cyclic code pass applied local K in MPa sqrt(m) to the same `sigma=K/sqrt(2*pi*r)` opening map; at R=0.1, `DeltaK=0.9*Kmax`. For the sole rising row, the highest prior qualified parent anchor at 24.3 MPa sqrt(m) is above `0.90*K_fracture_300=2.94544 MPa sqrt(m)`. The frozen sparse-load rule therefore returns `NO_UNSAMPLED_SUBFRACTURE_FATIGUE_INTERVAL`; no new fatigue load is legal.
