from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.lib.io import make_run_dirs, write_json
from src.steps.step1_rg160 import generate_target_spectra_xlsx
from src.steps.step2_take_spec import split_target_xlsx_to_txt
from src.steps.step3_eq_gen import generate_ground_motions
from src.steps.step4_sa_plot import export_psa
from src.steps.step5_collect import build_dataset


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _fmt_yb_for_run_id(yb: float) -> str:
    """
    run-id에 '.' 들어가면 애매한 환경이 있어서 안전하게 변환:
      1.2 -> 1p2
      0.80 -> 0p8
    """
    s = f"{yb:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _get_yb_list(cfg: Dict[str, Any]) -> List[float]:
    """
    YAML에서 y_b_list가 있으면 리스트로 반환.
    없으면 단일 y_b(또는 기본 1.0)로 1회 실행.
    """
    step1 = cfg.get("step1_rg160", {})
    if "y_b_list" in step1:
        ybs = step1["y_b_list"]
        if not isinstance(ybs, list) or len(ybs) == 0:
            raise ValueError("step1_rg160.y_b_list must be a non-empty list")
        return [float(v) for v in ybs]

    # fallback: single y_b
    yb = float(step1.get("y_b", 1.0))
    return [yb]


def main():
    ap = argparse.ArgumentParser(description="Run-based training data pipeline")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--run-id", type=str, required=True, help="Run folder name under runs/")
    ap.add_argument("--project-root", type=str, default=".", help="Project root (default: .)")
    ap.add_argument("--nproc", type=int, default=None, help="Override worker processes")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)

    # 공통 입력(변하지 않는 것들)은 loop 밖에서 1번만 resolve
    seed_dir = (project_root / cfg["paths"]["seed_dir"]).resolve()
    nproc = args.nproc if args.nproc is not None else cfg["run"].get("nproc", None)

    yb_list = _get_yb_list(cfg)
    print(f"[INFO] y_b runs = {yb_list}")

    for y_b in yb_list:
        run_id_yb = f"{args.run_id}_yb_{_fmt_yb_for_run_id(y_b)}"
        run_paths = make_run_dirs(project_root, run_id_yb)

        # 매 반복마다 cfg 복사 후 y_b만 주입 (원본 cfg 오염 방지)
        cfg_local = deepcopy(cfg)
        cfg_local.setdefault("step1_rg160", {})
        cfg_local["step1_rg160"]["y_b"] = float(y_b)

        # snapshot config: 원본 YAML 그대로 저장
        run_paths.config_snapshot.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
        # 주입된 값도 한 번에 확인 가능하게 별도 기록
        (run_paths.root / "yb_value.txt").write_text(str(y_b), encoding="utf-8")

        # ---------------------------
        # Step1: rg160 -> Target-spectrums.xlsx
        # ---------------------------
        target_xlsx = run_paths.step1_dir / "Target-spectrums.xlsx"
        manifest1 = generate_target_spectra_xlsx(
            cfg_local["step1_rg160"],
            target_xlsx,
            make_plots=bool(cfg_local["run"].get("make_plots", False)),
        )
        write_json(run_paths.step1_dir / "manifest.json", manifest1)

        # ---------------------------
        # Step2: take_spec -> target_spectrum/*.txt
        # ---------------------------
        manifest2 = split_target_xlsx_to_txt(
            cfg_local["step2_take_spec"],
            target_xlsx,
            run_paths.step2_dir,
        )
        write_json(run_paths.root / "02_target_txt" / "manifest.json", manifest2)

        # ---------------------------
        # Step3: EQ_Gen -> gm matched txt
        # ---------------------------
        manifest3 = generate_ground_motions(
            cfg_local["step3_eq_gen"],
            run_paths.step2_dir,
            seed_dir,
            run_paths.step3_dir,
            nproc=nproc,
        )
        write_json(run_paths.step3_dir / "manifest.json", manifest3)

        # ---------------------------
        # Step4: SA_PLOT -> PSA xlsx (default)
        # ---------------------------
        manifest4 = export_psa(
            cfg_local["step4_sa_plot"],
            run_paths.step3_dir,
            run_paths.step4_dir,
            nproc=nproc,
        )
        write_json(run_paths.step4_dir / "manifest.json", manifest4)

        # ---------------------------
        # Step5: collect_spec -> dataset csv
        # ---------------------------
        # NOTE: step5 expects xlsx format by default.
        # If you export npz in step4, update step5 accordingly.
        manifest5 = build_dataset(
            cfg_local["step5_collect"],
            run_paths.step4_dir,
            run_paths.step5_dir,
        )
        write_json(run_paths.step5_dir / "manifest.json", manifest5)

        print(f"\n[OK] Finished y_b={y_b} -> {run_paths.root}")

    print("\n[OK] All y_b runs finished.")


if __name__ == "__main__":
    main()