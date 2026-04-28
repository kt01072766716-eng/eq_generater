from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def safe_filename(name: str) -> str:
    """Remove Windows-forbidden characters."""
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class RunPaths:
    root: Path
    config_snapshot: Path
    step1_dir: Path
    step2_dir: Path
    step3_dir: Path
    step4_dir: Path
    step5_dir: Path


def make_run_dirs(project_root: Path, run_id: str) -> RunPaths:
    runs_root = ensure_dir(project_root / "runs")
    root = ensure_dir(runs_root / run_id)

    step1_dir = ensure_dir(root / "01_target_xlsx")
    step2_dir = ensure_dir(root / "02_target_txt" / "target_spectrum")
    step3_dir = ensure_dir(root / "03_gm_matched")
    step4_dir = ensure_dir(root / "04_psa")
    step5_dir = ensure_dir(root / "05_dataset")

    config_snapshot = root / "00_config_snapshot.yaml"

    return RunPaths(
        root=root,
        config_snapshot=config_snapshot,
        step1_dir=step1_dir,
        step2_dir=step2_dir,
        step3_dir=step3_dir,
        step4_dir=step4_dir,
        step5_dir=step5_dir,
    )
