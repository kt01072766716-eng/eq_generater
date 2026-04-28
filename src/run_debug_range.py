from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from src.lib.io import write_json
from src.steps.step1_rg160 import generate_target_spectra
from src.steps.step2_take_spec import split_xlsx_to_txt
from src.steps.step3_eq_gen import generate_ground_motions
from src.steps.step4_sa_plot import export_psa
from src.steps.step5_collect import build_dataset


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/default.yaml")
    ap.add_argument("--run-id", required=True, help="existing run folder name under runs/")
    ap.add_argument("--start-step", type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--end-step", type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--project-root", default=".", help="project root")
    ap.add_argument("--nproc", type=int, default=None)
    args = ap.parse_args()

    if args.start_step > args.end_step:
        raise ValueError("start-step must be <= end-step")

    project_root = Path(args.project_root).resolve()
    cfg = load_yaml(Path(args.config).resolve())

    run_root = project_root / "runs" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    nproc = args.nproc if args.nproc is not None else cfg["run"].get("nproc", None)

    # step paths
    step1_dir = run_root / "01_target_xlsx"
    step2_dir = run_root / "02_target_txt"
    step3_dir = run_root / "03_gm_matched"
    step4_dir = run_root / "04_psa"
    step5_dir = run_root / "05_dataset"

    print(f"[INFO] run_root   : {run_root}")
    print(f"[INFO] start_step : {args.start_step}")
    print(f"[INFO] end_step   : {args.end_step}")
    print("-" * 80)

    # -------------------------
    # STEP 1
    # -------------------------
    if args.start_step <= 1 <= args.end_step:
        print("[RUN] STEP 1 - generate target spectra")
        manifest1 = generate_target_spectra(
            cfg["step1_rg160"],
            step1_dir,
        )
        write_json(step1_dir / "manifest.json", manifest1)
        print(f"[OK ] STEP 1 -> {step1_dir}")
        print("-" * 80)

    # -------------------------
    # STEP 2
    # -------------------------
    if args.start_step <= 2 <= args.end_step:
        print("[RUN] STEP 2 - split xlsx to txt")
        ensure_exists(step1_dir, "STEP1 output folder")
        manifest2 = split_xlsx_to_txt(
            cfg["step2_take_spec"],
            step1_dir,
            step2_dir,
        )
        write_json(step2_dir / "manifest.json", manifest2)
        print(f"[OK ] STEP 2 -> {step2_dir}")
        print("-" * 80)

    # -------------------------
    # STEP 3
    # -------------------------
    if args.start_step <= 3 <= args.end_step:
        print("[RUN] STEP 3 - generate ground motions")
        ensure_exists(step2_dir, "STEP2 output folder")

        seed_dir = project_root / cfg["paths"]["seed_dir"]
        ensure_exists(seed_dir, "seed_dir")

        manifest3 = generate_ground_motions(
            cfg["step3_eq_gen"],
            step2_dir,
            seed_dir,
            step3_dir,
            nproc=nproc,
        )
        write_json(step3_dir / "manifest.json", manifest3)
        print(f"[OK ] STEP 3 -> {step3_dir}")
        print("-" * 80)

    # -------------------------
    # STEP 4
    # -------------------------
    if args.start_step <= 4 <= args.end_step:
        print("[RUN] STEP 4 - export PSA")
        ensure_exists(step3_dir, "STEP3 output folder")

        manifest4 = export_psa(
            cfg["step4_sa_plot"],
            step3_dir,
            step4_dir,
            nproc=nproc,
        )
        write_json(step4_dir / "manifest.json", manifest4)
        print(f"[OK ] STEP 4 -> {step4_dir}")
        print("-" * 80)

    # -------------------------
    # STEP 5
    # -------------------------
    if args.start_step <= 5 <= args.end_step:
        print("[RUN] STEP 5 - build dataset")
        ensure_exists(step4_dir, "STEP4 output folder")

        manifest5 = build_dataset(
            cfg["step5_collect"],
            step4_dir,
            step5_dir,
        )
        write_json(step5_dir / "manifest.json", manifest5)
        print(f"[OK ] STEP 5 -> {step5_dir}")
        print("-" * 80)

    print("[DONE] requested step range finished successfully.")


if __name__ == "__main__":
    main()