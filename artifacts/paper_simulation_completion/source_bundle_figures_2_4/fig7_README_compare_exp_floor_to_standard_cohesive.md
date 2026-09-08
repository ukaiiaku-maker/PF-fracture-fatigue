# EXP-floor vs standard cohesive-law comparison

Run in MATLAB:

```matlab
compare_exp_floor_to_standard_cohesive
```

At the top of the script choose:

```matlab
example_class = 'peak';   % ceramic, peak, weakT, or DBTT
T_K = 900;
```

The script:

1. evaluates the selected tuned crack-opening EXP-floor barrier at the chosen temperature;
2. computes its activation volume from `v* = -dG/dsigma`;
3. fits bilinear, exponential, and polynomial cohesive traction-separation laws to reproduce the barrier and activation-volume response;
4. compares traction-separation shape, barrier-versus-stress, activation-volume-versus-stress, and Arrhenius velocity response;
5. saves a 4-panel PNG, editable MATLAB `.fig`, fitted parameters CSV, and comparison-curves CSV.

Outputs are written to:

```text
cohesive_exp_floor_comparison/
```

The default example is the final tuned `peak` crack-opening barrier at 900 K.
