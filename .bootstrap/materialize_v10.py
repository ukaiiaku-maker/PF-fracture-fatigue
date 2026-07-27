#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path.cwd()
BOOTSTRAP = ROOT / ".bootstrap"
SOURCE = Path("/tmp/arrhenius_source_v10")
SOURCE_REPO = "https://github.com/ukaiiaku-maker/Arrhenius_FEM_CZM_MPZ.git"
SOURCE_COMMIT = "f2cc107793ee0a194931b94b15b0f95f44db46bc"
OVERLAY_SHA256 = "316d13516588744bdde936a0ccf04954a6a83fc9b750ae33802cac20d5862958"
BASE_FILES = [
    "config.py", "materials.py", "mesh.py", "fem.py", "j_integral.py",
    "crystal.py", "plasticity.py", "sharp_front.py", "fatigue_postprocess.py",
]


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(list(args), cwd=cwd, check=True)


def clone_source() -> Path:
    override = BOOTSTRAP / "source_override"
    if override.exists():
        return override
    if SOURCE.exists():
        shutil.rmtree(SOURCE)
    run("git", "init", str(SOURCE))
    run("git", "remote", "add", "origin", SOURCE_REPO, cwd=SOURCE)
    run("git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT, cwd=SOURCE)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=SOURCE)
    resolved = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
    ).strip()
    if resolved != SOURCE_COMMIT:
        raise RuntimeError(f"source commit mismatch: {resolved}")
    return SOURCE / "arrhenius_fracture"


def reconstruct_overlay() -> Path:
    archive = Path("/tmp/v10_custom.tar.gz")
    with archive.open("wb") as out:
        for index in range(8):
            path = BOOTSTRAP / f"v10overlay.part_{index:02d}"
            if not path.exists():
                raise FileNotFoundError(path)
            out.write(base64.b64decode(path.read_text().strip(), validate=True))
        for index in range(8, 14):
            path = BOOTSTRAP / f"v10overlay.bin_{index:02d}"
            if not path.exists():
                raise FileNotFoundError(path)
            out.write(path.read_bytes())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != OVERLAY_SHA256:
        raise RuntimeError(
            f"overlay checksum mismatch: expected {OVERLAY_SHA256}, got {digest}"
        )
    return archive


def overlay_custom() -> None:
    archive = reconstruct_overlay()
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        root = ROOT.resolve()
        for member in members:
            target = (ROOT / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe archive path: {member.name}")
        tf.extractall(ROOT)


def patch_sharp_front() -> None:
    path = ROOT / "arrhenius_fracture" / "sharp_front.py"
    text = path.read_text()
    imports = (
        "from .material_manifest import MaterialManifest, default_manifest_path\n"
        "from .unified_mpz import MPZConfig\n"
        "from .unified_front import UnifiedMPZFrontEngine\n"
    )
    if "from .material_manifest import MaterialManifest" not in text:
        needle = "from .materials import PlasticityModel\n"
        if needle not in text:
            raise RuntimeError("sharp_front import insertion point missing")
        text = text.replace(needle, needle + imports, 1)

    build = '''def build_engine(args, mat) -> FrontEngine:
    f = FrontConfig()
    f.r0 = args.r_pz
    f.sigma_cap = args.sigma_cap_GPa * 1e9
    f.m_hits = args.multihit_m
    f.tau_c = args.multihit_tau
    f.nu0_c = args.nu0_cleave
    f.nu0_e = args.nu0_emit
    f.beta_back = args.beta_back
    f.c_blunt = args.c_blunt
    f.L_pz = args.L_pz
    f.v_emb_b3 = args.v_emb_b3
    f.wake_retain = args.wake_retain
    f.chi_shield = getattr(args, 'chi_shield', 0.0)
    f.emb_sat_frac = getattr(args, 'emb_sat_frac', 1.0)
    f.N_sat = getattr(args, 'N_sat', float('inf'))
    f.recover_k = getattr(args, 'recover_k', 0.0)
    f.v_rayleigh = getattr(args, 'v_rayleigh', float('inf'))
    f.max_advances_per_step = 1
    f.dN_cap = float('inf')
    f.da = args.da
    if getattr(args, 'rho0', None) is not None:
        f.rho0 = float(args.rho0)

    cb = apply_cleavage_barrier_args(default_cleavage_barrier(), args)
    eb = default_emission_barrier(mat.b)
    if getattr(args, 'emit_H0_eV', None) is not None:
        eb.H0_eV = args.emit_H0_eV

    material_manifest = getattr(args, 'material_manifest', None)
    material_class = getattr(args, 'material_class', None)
    if material_manifest or material_class:
        manifest_path = material_manifest or default_manifest_path(material_class)
        manifest = MaterialManifest.from_csv(manifest_path)
        mpz_cfg = MPZConfig(
            length_m=float(getattr(args, 'mpz_length_um', 100.0)) * 1.0e-6,
            n_bins=int(getattr(args, 'mpz_n_bins', 200)),
            source_bin_count=int(getattr(args, 'mpz_source_bins', 2)),
            blunting_length_m=float(getattr(args, 'mpz_blunting_length_um', 0.5)) * 1.0e-6,
            wake_length_m=float(getattr(args, 'wake_length_um', 100.0)) * 1.0e-6,
            wake_n_bins=int(getattr(args, 'wake_n_bins', 0)),
            wake_shielding=bool(getattr(args, 'wake_shielding', True)),
            wake_shield_projection=float(getattr(args, 'wake_shield_projection', 1.0)),
        )
        f.L_pz = mpz_cfg.length_m
        return UnifiedMPZFrontEngine(
            f, cb, eb, mat.G, mat.nu, mat.b, manifest, mpz_cfg
        )
    return FrontEngine(f, cb, eb, mat.G, mat.nu, mat.b)
'''
    pattern = re.compile(
        r"def build_engine\(args, mat\) -> FrontEngine:\n.*?\n"
        r"(?=# ----------------------------------------------------------------------------\n# 1D driver)",
        re.S,
    )
    text, count = pattern.subn(build + "\n", text, count=1)
    if count != 1:
        raise RuntimeError(f"expected one build_engine replacement, found {count}")

    parser_block = '''    # v10 unified material/state selection
    p.add_argument('--material-class', choices=['ceramic', 'weakT', 'DBTT'], default=None,
                   help='Use the promoted unified MPZ material class.')
    p.add_argument('--material-manifest', default=None)
    p.add_argument('--mpz-length-um', type=float, default=100.0)
    p.add_argument('--mpz-n-bins', type=int, default=200)
    p.add_argument('--mpz-source-bins', type=int, default=2)
    p.add_argument('--mpz-blunting-length-um', type=float, default=0.5)
    p.add_argument('--wake-length-um', type=float, default=100.0)
    p.add_argument('--wake-n-bins', type=int, default=0)
    p.add_argument('--wake-shielding', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--wake-shield-projection', type=float, default=1.0)

'''
    if "p.add_argument('--material-class'" not in text:
        marker = "    # 1d\n"
        if marker not in text:
            raise RuntimeError("sharp_front parser insertion point missing")
        text = text.replace(marker, parser_block + marker, 1)
    path.write_text(text)


def main() -> None:
    source = clone_source()
    package = ROOT / "arrhenius_fracture"
    package.mkdir(exist_ok=True)
    for name in BASE_FILES:
        shutil.copy2(source / name, package / name)
    overlay_custom()
    patch_sharp_front()
    run(sys.executable, "-m", "compileall", "-q", "arrhenius_fracture")


if __name__ == "__main__":
    main()
