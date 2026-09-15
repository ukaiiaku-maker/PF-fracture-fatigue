# V2 2-D single-crack 300 K / 1200 K decision

Campaign classification: `V2_NAMED_PARAMETERIZATION_SINGLE_CRACK_SPATIAL_TRANSFER_TEST`.

All 12 exact-row cases reached the 1000 µm projected-extension target with one active front, the frozen common seed, and the canonical 1x loading contract. The result demonstrates numerical completion of this spatial-transfer test; it does not establish experimental calibration, ASTM toughness, a conventional R-curve, or spatial branching behavior.

The reported K histories are `PF_MODEL_NATIVE_KJ_DRIVING_TRAJECTORY`. Reload-separated points are `RELOAD_SEPARATED_EFFECTIVE_RESISTANCE_CANDIDATES`; consecutive event-bearing accepted steps are treated as one uninterrupted avalanche and are not exported as independent resistance points. Applied/specimen K and global G/VCCT remain unavailable because this source tree contains no separately qualified canonical operator for them.

## P25/P40 comparison

- DBTT at 300 K: terminal P25/P40 KJ = 9.22405/12.1841 MPa√m (P40/P25 = 1.32091).
- DBTT at 1200 K: terminal P25/P40 KJ = 45.8559/110.824 MPa√m (P40/P25 = 2.41679).
- weakT at 300 K: terminal P25/P40 KJ = 9.96691/13.208 MPa√m (P40/P25 = 1.32518).
- weakT at 1200 K: terminal P25/P40 KJ = 10.7768/15.2908 MPa√m (P40/P25 = 1.41887).
- ceramic at 300 K: terminal P25/P40 KJ = 9.3301/12.478 MPa√m (P40/P25 = 1.33739).
- ceramic at 1200 K: terminal P25/P40 KJ = 3.50734/4.82718 MPa√m (P40/P25 = 1.37631).

## Terminal dispositions

- `DBTT_V2_300K_theta0`: `V2_SINGLE_CRACK_2D_PF_REACHED_1000UM`; 1000.01 µm projected, 1003.63 µm path, 207 events in 207 physical avalanches, terminal model-native KJ 9.22405 MPa√m.
- `DBTT_V2_1200K_theta0`: `V2_SINGLE_CRACK_2D_PF_REACHED_1000UM`; 1000.01 µm projected, 1003.63 µm path, 207 events in 177 physical avalanches, terminal model-native KJ 45.8559 MPa√m.
- `DBTT_V2_P40_300K_theta0`: `V2_SINGLE_CRACK_2D_PF_REACHED_1000UM`; 1000.01 µm projected, 1003.63 µm path, 207 events in 207 physical avalanches, terminal model-native KJ 12.1841 MPa√m.
- `DBTT_V2_P40_1200K_theta0`: `V2_SINGLE_CRACK_2D_PF_REACHED_1000UM`; 1000.01 µm projected, 1003.63 µm path, 207 events in 207 physical avalanches, terminal model-native KJ 110.824 MPa√m.
- `weakT_V2_P25_300K_theta0`: `V2_SINGLE_CRACK_2D_PF_REACHED_1000UM`; 1000.01 µm projected, 1003.63 µm path, 207 events in 207 physical avalanches, terminal model-native KJ 9.96691 MPa√m.
- `weakT_V2_P25_1200K_theta0`: `V2_SINGLE_CRACK_2D_PF_REACHED_1000UM`; 1000.01 µm projected, 1003.63 µm path, 207 events in 207 physical avalanches, terminal model-native KJ 10.7768 MPa√m.
- `weakT_V2_P40_300K_theta0`: `V2_SINGLE_CRACK_2D_PF_REACHED_1000UM`; 1000.01 µm projected, 1003.63 µm path, 207 events in 207 physical avalanches, terminal model-native KJ 13.208 MPa√m.
- `weakT_V2_P40_1200K_theta0`: `V2_SINGLE_CRACK_2D_PF_REACHED_1000UM`; 1000.01 µm projected, 1003.63 µm path, 207 events in 207 physical avalanches, terminal model-native KJ 15.2908 MPa√m.
- `ceramic_V2_P25_300K_theta0`: `V2_SINGLE_CRACK_2D_PF_REACHED_1000UM`; 1000.01 µm projected, 1003.63 µm path, 207 events in 207 physical avalanches, terminal model-native KJ 9.3301 MPa√m.
- `ceramic_V2_P25_1200K_theta0`: `V2_SINGLE_CRACK_2D_PF_REACHED_1000UM`; 1000.01 µm projected, 1003.63 µm path, 207 events in 207 physical avalanches, terminal model-native KJ 3.50734 MPa√m.
- `ceramic_V2_P40_300K_theta0`: `V2_SINGLE_CRACK_2D_PF_REACHED_1000UM`; 1000.01 µm projected, 1003.63 µm path, 207 events in 207 physical avalanches, terminal model-native KJ 12.478 MPa√m.
- `ceramic_V2_P40_1200K_theta0`: `V2_SINGLE_CRACK_2D_PF_REACHED_1000UM`; 1000.01 µm projected, 1003.63 µm path, 207 events in 194 physical avalanches, terminal model-native KJ 4.82718 MPa√m.
