from __future__ import annotations

import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# NOTE: reqpy_M must be importable in your environment.
from reqpy_M import REQPY_single


def make_perturbed_target(dso, rng, sigma=0.05, smooth_window=5, clip_min=0.75, clip_max=1.25):
    n = len(dso)
    noise = rng.normal(0.0, sigma, size=n)
    if smooth_window and smooth_window > 1:
        k = np.ones(int(smooth_window)) / float(smooth_window)
        noise = np.convolve(noise, k, mode="same")
    factor = 1.0 + noise
    factor = np.clip(factor, clip_min, clip_max)
    return dso * factor


def randomize_seed(base_seed, rng, noise_std_ratio=0.09, max_shift=50, spectral_perturb_ratio=0.10, keep_rms=True):
    x = base_seed.astype(float).copy()
    N = len(x)

    if noise_std_ratio and noise_std_ratio > 0:
        std = np.std(x) + 1e-12
        x += rng.normal(0.0, float(noise_std_ratio) * std, size=N)

    if max_shift and int(max_shift) > 0:
        shift = rng.integers(-int(max_shift), int(max_shift) + 1)
        x = np.roll(x, shift)

    if spectral_perturb_ratio and spectral_perturb_ratio > 0:
        X = np.fft.rfft(x)
        gain = 1.0 + float(spectral_perturb_ratio) * rng.normal(0.0, 1.0, size=len(X))
        x = np.fft.irfft(X * gain, n=N)

    if keep_rms:
        rms0 = np.sqrt(np.mean(base_seed**2)) + 1e-12
        rms1 = np.sqrt(np.mean(x**2)) + 1e-12
        x = x * (rms0 / rms1)

    return x


def _stable_int_seed(base_seed: int, spectrum_name: str) -> int:
    # reproducible across runs and platforms
    h = 2166136261
    for ch in spectrum_name.encode("utf-8"):
        h = (h ^ ch) * 16777619
        h &= 0xFFFFFFFF
    return int((h ^ int(base_seed)) & 0xFFFFFFFF)


def _parse_target_params_from_name(spectrum_name: str, param_names: List[str], separator: str) -> Dict[str, float]:
    parts = spectrum_name.split(separator)
    if len(parts) != len(param_names):
        raise ValueError(
            f"Target spectrum name '{spectrum_name}' must have {len(param_names)} fields split by '{separator}', "
            f"got {len(parts)} fields: {parts}"
        )

    values: Dict[str, float] = {}
    for key, raw in zip(param_names, parts):
        try:
            values[str(key)] = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"Target spectrum name '{spectrum_name}' contains non-numeric field '{raw}' for key '{key}'"
            ) from exc
    return values


def _value_in_interval(value: float, rule: Dict[str, Any]) -> bool:
    min_v = rule.get("min", None)
    max_v = rule.get("max", None)
    include_min = bool(rule.get("include_min", True))
    include_max = bool(rule.get("include_max", False))

    if min_v is not None:
        min_v = float(min_v)
        if include_min:
            if value < min_v:
                return False
        else:
            if value <= min_v:
                return False

    if max_v is not None:
        max_v = float(max_v)
        if include_max:
            if value > max_v:
                return False
        else:
            if value >= max_v:
                return False

    return True


def _matches_type_conditions(values: Dict[str, float], type_rule: Dict[str, Any]) -> bool:
    """
    values:
        파일명에서 파싱된 원래 파라미터 값들
        예) {"b": ..., "y_b": ..., "c": ..., "y_c": ...}

    지원 조건 키:
        - 원래 값: b, c, y_b, y_c ...
        - 파생 값: cb_gap = c - b
    """
    check_values = dict(values)

    # 파생 변수 추가
    if "b" in values and "c" in values:
        check_values["cb_gap"] = values["c"] - values["b"]

    for key, rule in type_rule.items():
        if key not in check_values:
            raise KeyError(
                f"Type condition key '{key}' not found in parsed target params or derived params: "
                f"{list(check_values.keys())}"
            )

        if not isinstance(rule, dict):
            raise TypeError(
                f"Condition for '{key}' must be a mapping with "
                f"min/max/include_min/include_max"
            )

        if not _value_in_interval(check_values[key], rule):
            return False

    return True


def _select_seed_files_for_target(target_path: Path, seed_dir: Path, cfg: Dict[str, Any]) -> Tuple[List[Path], str | None]:
    selector_cfg = cfg.get("seed_type_selector", {}) or {}
    type_conditions = selector_cfg.get("type_conditions", {}) or {}

    # Backward compatible: if no conditions are defined, use all txt files directly under seed_dir.
    if not type_conditions:
        seed_files = sorted(seed_dir.glob("*.txt"))
        return seed_files, None

    param_names = [str(v) for v in selector_cfg.get("param_names", ["a", "b", "c", "d"])]
    separator = str(selector_cfg.get("separator", "l"))

    target_values = _parse_target_params_from_name(target_path.stem, param_names, separator)

    matched_types: List[str] = []
    for type_name, type_rule in type_conditions.items():
        if _matches_type_conditions(target_values, type_rule):
            matched_types.append(str(type_name))
    '''
    if not matched_types:
        raise RuntimeError(
            f"No seed type matched target '{target_path.name}' with parsed values {target_values}. "
            f"Check step3_eq_gen.seed_type_selector.type_conditions in YAML."
        )
    '''
    if not matched_types:
        # fallback: use all seeds
        seed_files = sorted(seed_dir.rglob("*.txt"))

        return seed_files, None
    if len(matched_types) > 1:
        raise RuntimeError(
            f"Multiple seed types matched target '{target_path.name}': {matched_types}. "
            f"Make the YAML conditions mutually exclusive."
        )

    seed_type = matched_types[0]
    type_dir = seed_dir / seed_type
    if not type_dir.is_dir():
        raise RuntimeError(f"Matched seed type '{seed_type}' but folder does not exist: {type_dir}")

    seed_files = sorted(type_dir.glob("*.txt"))
    return seed_files, seed_type


def _process_one_spectrum(job: Tuple[str, str, Dict[str, Any]]):
    target_path_str, seed_dir_str, cfg = job
    target_path = Path(target_path_str)
    seed_dir = Path(seed_dir_str)

    spectrum_name = target_path.stem
    out_root = Path(cfg["out_root"])
    output_dir = out_root / spectrum_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()

    base_seed = int(cfg["reproducibility"]["base_seed"])
    rng = np.random.default_rng(_stable_int_seed(base_seed, spectrum_name))

    seed_files, matched_seed_type = _select_seed_files_for_target(target_path, seed_dir, cfg)
    if not seed_files:
        if matched_seed_type is None:
            raise RuntimeError(f"No seed txt in: {seed_dir}")
        raise RuntimeError(f"No seed txt in matched seed folder: {seed_dir / matched_seed_type}")

    # read target
    spec = np.loadtxt(target_path)
    T_target = spec[:, 0]
    Sa_target = spec[:, 1]

    N_GM = int(cfg["N_GM"])
    dt = float(cfg["dt"])
    fs = 1.0 / dt

    reqpy_cfg = cfg["reqpy"]
    targ_p = cfg["target_perturb"]
    seed_p = cfg["seed_perturb"]

    # -----------------------------
    # seed 비복원 셔플 순환 준비
    # -----------------------------
    seed_files_cycle = list(seed_files)
    rng.shuffle(seed_files_cycle)
    seed_idx = 0

    for k in range(1, N_GM + 1):
        gm_name = f"GM_multi_{k:02d}"
        if k == 1:
            if matched_seed_type is None:
                print(f"[PID {pid}] spectrum={spectrum_name}  N_GM={N_GM}  seed_pool=all")
            else:
                print(f"[PID {pid}] spectrum={spectrum_name}  N_GM={N_GM}  seed_type={matched_seed_type}")

        # -----------------------------------------
        # 한 바퀴 다 쓰면 다시 섞고 처음부터 사용
        # -----------------------------------------
        if seed_idx >= len(seed_files_cycle):
            rng.shuffle(seed_files_cycle)
            seed_idx = 0

        seed_file = seed_files_cycle[seed_idx]
        seed_idx += 1

        # 필요하면 디버그 출력
        # print(f"[DEBUG] spectrum={spectrum_name}, GM={k:02d}, seed={seed_file.name}")

        base_seed_arr = np.loadtxt(seed_file)

        seed_rand = randomize_seed(base_seed_arr, rng, **seed_p)
        Sa_pert = make_perturbed_target(Sa_target, rng, **targ_p)

        results = REQPY_single(
            s=seed_rand,
            fs=fs,
            dso=Sa_pert,
            To=T_target,
            T1=float(reqpy_cfg["T1"]),
            T2=float(reqpy_cfg["T2"]),
            zi=float(cfg.get("zeta", 0.05)),
            nit=int(reqpy_cfg["nit"]),
        )
        matched = results["ccs"]
        out_txt = output_dir / f"{gm_name}_matched.txt"
        np.savetxt(out_txt, matched, fmt="%.6e")

    return spectrum_name

def generate_ground_motions(cfg: Dict[str, Any], target_dir: Path, seed_dir: Path, out_root: Path, nproc: int | None = None) -> Dict[str, Any]:
    target_files = sorted(target_dir.glob("*.txt"))
    if not target_files:
        raise RuntimeError(f"No target spectra txt in: {target_dir}")

    selector_cfg = cfg.get("seed_type_selector", {}) or {}
    if selector_cfg.get("type_conditions"):
        if not seed_dir.is_dir():
            raise RuntimeError(f"Seed directory not found: {seed_dir}")
    else:
        seed_files = sorted(seed_dir.glob("*.txt"))
        if not seed_files:
            raise RuntimeError(f"No seed txt in: {seed_dir}")

    out_root.mkdir(parents=True, exist_ok=True)

    cfg2 = dict(cfg)
    cfg2["out_root"] = str(out_root)

    jobs = [(str(tp), str(seed_dir), cfg2) for tp in target_files]

    if nproc is None:
        nproc = min(cpu_count(), len(target_files))
    else:
        nproc = max(1, min(int(nproc), len(target_files)))

    with Pool(processes=nproc) as pool:
        done = pool.map(_process_one_spectrum, jobs)

    return {"num_spectra": len(done), "spectra": done, "out_root": str(out_root)}
