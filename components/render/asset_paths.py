"""
SIM1 render pipeline imports: paths come from scripts.sim1_asset_paths (Hugging Face bundle).

This module prepends the repository root to sys.path so `import scripts.sim1_asset_paths`
works when Blender/scripts run with cwd under components/render/.
"""

from __future__ import annotations

import os
import sys

_RENDER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_RENDER_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.sim1_asset_paths import (  # noqa: E402
    describe_assets_resolution,
    get_acone_urdf_path,
    get_assets_root,
    get_bg_root,
    get_cloth_usdc_path,
    get_flow_ckpt_three_path,
    get_mat_root,
    get_project_root,
    get_random_assets_dir,
    get_render_assets_dir,
    get_table_root,
    validate_render_assets,
    validate_render_assets_or_exit,
)

__all__ = [
    "describe_assets_resolution",
    "get_acone_urdf_path",
    "get_assets_root",
    "get_bg_root",
    "get_cloth_usdc_path",
    "get_flow_ckpt_three_path",
    "get_mat_root",
    "get_project_root",
    "get_random_assets_dir",
    "get_render_assets_dir",
    "get_table_root",
    "validate_render_assets",
    "validate_render_assets_or_exit",
]
