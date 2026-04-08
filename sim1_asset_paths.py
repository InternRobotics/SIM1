"""
SIM1 Hugging Face asset layout — single source of truth for paths.

After `bash download_assets.sh`, files live under <repo>/assets/ by default.
Override with environment variable:

  SIM1_ASSETS_ROOT   Absolute or relative path to the folder that *contains*
                       acone/, cloth/, random/, model/, etc. (i.e. the directory you pass
                       to huggingface-cli / hf download as --local-dir).

Default HDRI / table / cloth glTF roots: <bundle>/random/{bg,table,mat}/.
Optional overrides: SIM1_BG_ROOT, SIM1_TABLE_ROOT, SIM1_MAT_ROOT

  Diffusion checkpoint (data generation):

    <bundle>/model/flow_ckpt_three.pth   — use get_flow_ckpt_three_path()
"""

from __future__ import annotations

import os
import sys

_SIM1_ROOT = os.path.dirname(os.path.abspath(__file__))
_ASSETS_ENV = "SIM1_ASSETS_ROOT"


def get_project_root() -> str:
    """SIM1 repository root (directory containing sim1_asset_paths.py)."""
    return _SIM1_ROOT


def get_assets_root() -> str:
    """
    Root directory for Hugging Face–downloaded bundles (acone, cloth, random, model, …).

    Default: <project_root>/assets
    Override: export SIM1_ASSETS_ROOT=/path/to/parent/of/acone
    """
    raw = os.environ.get(_ASSETS_ENV, "").strip()
    if raw:
        return os.path.normpath(os.path.join(get_project_root(), raw) if not os.path.isabs(raw) else raw)
    return os.path.join(get_project_root(), "assets")


def get_render_assets_dir() -> str:
    """Legacy layout: ``assets/render/`` (unused by default bg/table/mat; see ``get_random_assets_dir``)."""
    return os.path.join(get_assets_root(), "render")


def get_random_assets_dir() -> str:
    """Randomized scene assets from the HF bundle: ``assets/random/``."""
    return os.path.join(get_assets_root(), "random")


def get_acone_urdf_path() -> str:
    return os.path.join(get_assets_root(), "acone", "acone.urdf")


def get_cloth_usdc_path() -> str:
    return os.path.join(get_assets_root(), "cloth", "short-shirt.usdc")


def get_flow_ckpt_three_path() -> str:
    """Diffusion policy checkpoint shipped with the Hugging Face bundle (`assets/model/flow_ckpt_three.pth`)."""
    return os.path.join(get_assets_root(), "model", "flow_ckpt_three.pth")


def get_bg_root() -> str:
    return os.environ.get("SIM1_BG_ROOT", os.path.join(get_random_assets_dir(), "bg"))


def get_table_root() -> str:
    return os.environ.get("SIM1_TABLE_ROOT", os.path.join(get_random_assets_dir(), "table"))


def get_mat_root() -> str:
    return os.environ.get("SIM1_MAT_ROOT", os.path.join(get_random_assets_dir(), "mat"))


def describe_assets_resolution() -> str:
    """One-line hint for logs / errors."""
    return (
        f"SIM1 assets root = {get_assets_root()} "
        f"(set {_ASSETS_ENV} to the Hugging Face download directory; default is <repo>/assets)"
    )


def validate_render_assets() -> list[str]:
    """Human-readable problems; empty if layout looks OK for the render pipeline."""
    problems: list[str] = []
    urdf = get_acone_urdf_path()
    if not os.path.isfile(urdf):
        problems.append(f"Missing robot URDF (Step 2): {urdf}")

    for name, path in (
        ("bg (HDRI)", get_bg_root()),
        ("table (glTF)", get_table_root()),
        ("mat (cloth glTF)", get_mat_root()),
    ):
        if not os.path.isdir(path):
            problems.append(f"Missing {name} directory: {path}")

    return problems


def validate_render_assets_or_exit() -> None:
    problems = validate_render_assets()
    if not problems:
        print(f"[render] {describe_assets_resolution()}")
        return
    print(
        "[render] Asset check failed — run `bash download_assets.sh` or set "
        f"{_ASSETS_ENV} to your Hugging Face asset directory.",
        file=sys.stderr,
    )
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    root = get_assets_root()
    print(
        f"\nExpected layout under {_ASSETS_ENV} or <repo>/assets:\n"
        f"  {os.path.join(root, 'acone', 'acone.urdf')}\n"
        f"  {os.path.join(root, 'random', 'bg')}/          (.exr HDRI files)\n"
        f"  {os.path.join(root, 'random', 'table')}/       (<name>/<name>.gltf)\n"
        f"  {os.path.join(root, 'random', 'mat')}/         (<name>/<name>.gltf)\n",
        file=sys.stderr,
    )
    sys.exit(1)
