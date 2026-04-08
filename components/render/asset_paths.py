"""
SIM1 render pipeline: resolve all bundled assets under <project_root>/assets/.

Override the root with env SIM1_ASSETS_ROOT (e.g. after huggingface-cli download to a custom dir).
Per-category overrides: SIM1_BG_ROOT, SIM1_TABLE_ROOT, SIM1_MAT_ROOT (optional).
"""

from __future__ import annotations

import os
import sys

_RENDER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_RENDER_DIR))


def get_project_root() -> str:
    """Repository root (parent of components/)."""
    return _PROJECT_ROOT


def get_assets_root() -> str:
    """Root folder for HF assets; default <project_root>/assets."""
    return os.environ.get("SIM1_ASSETS_ROOT", os.path.join(_PROJECT_ROOT, "assets"))


def get_render_assets_dir() -> str:
    return os.path.join(get_assets_root(), "render")


def get_acone_urdf_path() -> str:
    return os.path.join(get_assets_root(), "acone", "acone.urdf")


def get_bg_root() -> str:
    return os.environ.get("SIM1_BG_ROOT", os.path.join(get_render_assets_dir(), "bg"))


def get_table_root() -> str:
    return os.environ.get("SIM1_TABLE_ROOT", os.path.join(get_render_assets_dir(), "table"))


def get_mat_root() -> str:
    return os.environ.get("SIM1_MAT_ROOT", os.path.join(get_render_assets_dir(), "mat"))


def validate_render_assets() -> list[str]:
    """
    Return a list of human-readable problems (empty if OK).
    Does not require bg/table/mat to be non-empty (Step3 fails clearly if empty).
    """
    problems: list[str] = []
    root = get_assets_root()
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
        print(f"[render] Assets root: {get_assets_root()}")
        return
    print("[render] Asset check failed — download SIM1 assets into the expected tree, or set SIM1_ASSETS_ROOT.", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    print(
        "\nExpected layout (under project root or SIM1_ASSETS_ROOT):\n"
        "  assets/acone/acone.urdf\n"
        "  assets/render/bg/          (.exr HDRI files)\n"
        "  assets/render/table/       (<name>/<name>.gltf)\n"
        "  assets/render/mat/         (<name>/<name>.gltf)\n",
        file=sys.stderr,
    )
    sys.exit(1)
