# v10.2.5 installation, test gate, and signed-dislocation guide

## Scope and present status

Branch: `v10.2.5-signed-burgers-shared-physics`

This revision implements one shared signed-dislocation state model for monotonic
temperature-dependent fracture and cyclic fatigue. The software architecture and
regression suite are operational. Production fracture/fatigue calculations remain
blocked until a real candidate-independent 2-D signed shielding kernel and a
mechanically derived source-normalization artifact have been generated.

The included software gate does **not** use a physical kernel and must not be
interpreted as validating toughness, fatigue growth, shielding magnitude, or a
material parameterization.

## Clean installation on another computer

### 1. System prerequisites

Install Git and a Conda-compatible Python distribution such as Miniforge,
Miniconda, or Anaconda. On macOS, install the command-line developer tools if Git
or a compiler is missing:

```bash
xcode-select --install
```

Verify the tools:

```bash
git --version
conda --version
```

### 2. Clone the exact development branch

Choose a parent directory and clone only the signed-physics branch:

```bash
cd /path/to/your/projects

git clone \
  --branch v10.2.5-signed-burgers-shared-physics \
  --single-branch \
  https://github.com/ukaiiaku-maker/PF-fracture-fatigue.git \
  PF-fracture-fatigue_v10_2_5_signed_burgers

cd PF-fracture-fatigue_v10_2_5_signed_burgers

git rev-parse --short HEAD
git status --short
```

The working tree should be clean. Record the printed commit because the branch may
advance during development.

### 3. Create the Python environment

The package requires Python 3.10 or newer. Python 3.12 is the validated local
configuration:

```bash
conda create -n arrhenius-sharp-front-v10 python=3.12 -y
conda activate arrhenius-sharp-front-v10

python --version
```

### 4. Install the repository and test dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[test]"
```

The declared numerical dependencies are NumPy, SciPy, and Matplotlib; pytest is
installed through the `test` optional dependency.

### 5. Compile and run the full regression suite

```bash
python -m compileall -q arrhenius_fracture scripts
python -m pytest -q
```

At commit `6765e84`, the expected result was `135 passed`. Later commits may add
additional tests, so the required condition is zero failures rather than an exact
count.

### 6. Run the signed-dislocation software gate

First update to the latest branch head:

```bash
git fetch origin \
  refs/heads/v10.2.5-signed-burgers-shared-physics:refs/remotes/origin/v10.2.5-signed-burgers-shared-physics

git merge --ff-only \
  origin/v10.2.5-signed-burgers-shared-physics
```

Then run the guarded gate using a new output directory:

```bash
OUTROOT=runs/v10_2_5_signed_software_gate_v1 \
FULL_SUITE=1 \
bash scripts/run_v10_2_5_signed_software_gate.sh
```

The gate checks compilation, signed-content cancellation, sign reversal,
antishielding, sign-preserving moving-frame transfer, source-capacity rejection,
30 GPa local-strength replay parity, common monotonic/fatigue engine installation,
and continued blocking of the invalid v10.2.4 campaign.

Expected final file:

```text
runs/v10_2_5_signed_software_gate_v1/software_gate.json
```

A passing software gate contains:

```json
{
  "physical_kernel_used": false,
  "production_fracture_physics_validated": false,
  "production_fatigue_physics_validated": false,
  "pass": true
}
```

These `false` fields are deliberate. They prevent a successful software test from
being mistaken for physical validation.

## Updating an existing clone

```bash
cd /path/to/PF-fracture-fatigue_v10_2_5_signed_burgers

git status --short

git fetch origin \
  refs/heads/v10.2.5-signed-burgers-shared-physics:refs/remotes/origin/v10.2.5-signed-burgers-shared-physics

git merge --ff-only \
  origin/v10.2.5-signed-burgers-shared-physics

python -m pip install -e ".[test]"
python -m compileall -q arrhenius_fracture scripts
python -m pytest -q
```

Do not merge with uncommitted local modifications. Save them on a separate branch
or commit before updating.

## Signed-dislocation treatment

### State variables

Each reduced slip channel carries positive and negative Burgers populations
separately in the active process zone and wake:

- mobile positive and negative content;
- retained positive and negative content;
- accumulated positive and negative slip;
- wake mobile, retained, and slip content for both signs.

Unsigned totals are the sums of the two signs and remain the appropriate variables
for forest density, Taylor backstress, recovery, trapping, transport, source
exhaustion, and blunting. The signed difference is used only where sign matters
mechanically.

For channel `s` and spatial bin `i`,

```text
N_signed[s,i] = N_positive[s,i] - N_negative[s,i]
```

### Signed shielding operator

The crack-tip shielding contribution is

```text
K_shield = sum(H_active[s,i] * N_signed_active[s,i])
         + sum(H_wake[s,i]   * N_signed_wake[s,i])
```

The engine uses

```text
K_tip = K_applied - K_shield
```

Therefore positive `K_shield` is shielding and negative `K_shield` is
antishielding. Reversing Burgers sign reverses the interaction automatically.
Equal positive and negative populations can contribute to density and backstress
while canceling in the signed shielding sum.

### Emission sign

The sign of newly emitted line content is selected from the signed resolved shear
reported by the 2-D anisotropic mechanics for each reduced channel. A zero or
unreliable signed drive cannot silently create unsigned content.

### Source sites versus line content

`source_sites_per_system` is a statistical count of nucleation opportunities. It
is not directly inserted into the shielding sum as a count of fully coherent
lines.

A mandatory mechanics-derived conversion maps source activations to physical line
content:

```text
delta_line_content[s] = delta_source_activations[s]
                      * activation_to_line_content_by_system[s]
```

The same artifact supplies physically admissible source-capacity bounds. Old
anchors with thousands of source sites are rejected when they lie outside those
bounds.

### Required 2-D kernel

For each active and wake bin, reduced slip channel, and Burgers sign, a fixed-crack
2-D unit perturbation calculation must provide a base and perturbed mode-I crack-tip
response. The shielding coefficient is defined with the engine sign convention:

```text
H[s,i] = (K_tip_base - K_tip_perturbed)
       / delta_signed_line_content
```

Both positive and negative perturbations are required. The builder checks their
normalized responses for linearity and antisymmetry and rejects incomplete
matrices. No default `(1,1)` projection, K-shield cap, or fitted attenuation is
permitted.

### Shared fracture and fatigue implementation

Both loading paths use:

```bash
python -m arrhenius_fracture.sharp_front_v10_2_5
```

Adding `--fatigue-cycles` selects cyclic loading, but both paths install the same
`SignedBurgersAnisotropicTipEngine`, source normalization, transport law,
moving-frame operator, and shielding kernel. This is the architectural constraint
that prevents monotonic fracture and fatigue from developing separate constitutive
implementations.

### Local strength limit

The approximately 30 GPa `sigma_cap` is retained as a local cohesive/strength
limit on the reconstructed crack-tip opening stress. It is not a cap on
`K_shield`. Exact replay copies this and every other serialized front, MPZ,
kinetic, anisotropic, transport, and campaign setting.

## What is not yet physically validated

Do not begin a parameter campaign until all of the following pass:

1. candidate-independent 2-D positive/negative unit perturbation responses;
2. mechanically derived activation-to-line normalization;
3. mechanically derived source-capacity bounds;
4. signed kernel construction and antisymmetry checks;
5. exact 2-D/replay configuration and final-state equivalence;
6. monotonic and cyclic smoke calculations using the same real kernel;
7. conservation and geometry-based rejection criteria.

A synthetic software gate is useful for implementation integrity but is not a
substitute for these mechanics calculations.
