from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_labels_from_stem(stem: str) -> Tuple[float, float, float, float]:
    parts = stem.split("l")
    if len(parts) != 4:
        raise ValueError(f"Filename stem must be y1ly2ly3ly4: {stem}")
    y = []
    for p in parts:
        p = p.strip().replace(",", ".")
        y.append(float(p))
    return tuple(y)  # type: ignore


def build_dataset_from_xlsx_folder(cfg: Dict[str, Any], input_dir: Path, out_dir: Path) -> Dict[str, Any]:
    xtol = float(cfg.get("xtol", 1e-9))
    out_dir.mkdir(parents=True, exist_ok=True)

    out_x = out_dir / str(cfg.get("output", {}).get("X_name", "input_f_train.csv"))
    out_y = out_dir / str(cfg.get("output", {}).get("y_name", "y_train.csv"))

    xlsx_files = sorted(input_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise RuntimeError(f"No .xlsx files in: {input_dir}")

    X_cols: List[np.ndarray] = []
    col_names: List[str] = []
    y_rows: List[List[float]] = []

    x_ref = None
    L_ref = None

    for fp in xlsx_files:
        y1, y2, y3, y4 = parse_labels_from_stem(fp.stem)
        sheets = pd.read_excel(fp, sheet_name=None, engine="openpyxl")

        for sheet_name, df in sheets.items():
            if df is None or df.shape[1] < 2:
                continue

            x = df.iloc[:, 0].to_numpy(dtype=float)
            F = df.iloc[:, 1:]

            if L_ref is None:
                L_ref = len(x)
            if len(x) != L_ref:
                raise ValueError(f"Length mismatch: {fp.name} / sheet={sheet_name}")

            if x_ref is None:
                x_ref = x.copy()
            else:
                if not np.allclose(x, x_ref, atol=xtol, rtol=0):
                    raise ValueError(f"x-grid mismatch: {fp.name} / sheet={sheet_name}")

            for j, col in enumerate(F.columns):
                f = F.iloc[:, j].to_numpy(dtype=np.float32)
                if not np.isfinite(f).all():
                    raise ValueError(f"NaN/inf in {fp.name} / sheet={sheet_name} / col={col}")

                cname = f"{fp.stem}__{sheet_name}__{col}"
                X_cols.append(f)
                col_names.append(cname)
                y_rows.append([y1, y2, y3, y4])

    if not X_cols:
        raise RuntimeError("No usable columns found")

    X_arr = np.stack(X_cols, axis=0)  # (nSamples, nPoints)
    X_df = pd.DataFrame(X_arr.T, columns=col_names)
    y_df = pd.DataFrame(y_rows, columns=["y1", "y2", "y3", "y4"])

    X_df.to_csv(out_x, index=False)
    y_df.to_csv(out_y, index=False)

    return {"X_shape": list(X_df.shape), "y_shape": list(y_df.shape), "out_x": str(out_x), "out_y": str(out_y)}


def build_dataset_from_npz_folder(cfg: Dict[str, Any], input_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """
    Step4(npz) 출력 포맷을 읽어서 dataset 생성.
    Step4가 저장한 npz 키:
      - T: (nT,)
      - Sa: (nGM, nT)
      - names: (nGM,)
    파일명(stem)에서 y1ly2ly3ly4 라벨 파싱은 기존과 동일.
    """
    xtol = float(cfg.get("xtol", 1e-9))
    out_dir.mkdir(parents=True, exist_ok=True)

    out_x = out_dir / str(cfg.get("output", {}).get("X_name", "input_f_train.csv"))
    out_y = out_dir / str(cfg.get("output", {}).get("y_name", "y_train.csv"))

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz files in: {input_dir}")

    X_cols: List[np.ndarray] = []
    col_names: List[str] = []
    y_rows: List[List[float]] = []

    T_ref = None
    nT_ref = None

    for fp in npz_files:
        y1, y2, y3, y4 = parse_labels_from_stem(fp.stem)

        data = np.load(fp, allow_pickle=True)
        if "T" not in data or "Sa" not in data:
            raise ValueError(f"Missing keys in npz: {fp.name} (need T, Sa)")

        T = np.asarray(data["T"], dtype=float).reshape(-1)  # (nT,)
        Sa = np.asarray(data["Sa"], dtype=np.float32)        # (nGM, nT) expected

        if Sa.ndim != 2:
            raise ValueError(f"Sa must be 2D (nGM, nT): {fp.name}")

        if nT_ref is None:
            nT_ref = len(T)
        if len(T) != nT_ref:
            raise ValueError(f"T length mismatch: {fp.name} (got {len(T)}, ref {nT_ref})")

        if T_ref is None:
            T_ref = T.copy()
        else:
            if not np.allclose(T, T_ref, atol=xtol, rtol=0):
                raise ValueError(f"T-grid mismatch: {fp.name}")

        # names는 없을 수도 있어서 방어
        if "names" in data:
            names = np.asarray(data["names"])
        else:
            names = np.array([f"gm_{i:03d}" for i in range(Sa.shape[0])], dtype=object)

        if Sa.shape[0] != len(names):
            raise ValueError(f"names length mismatch: {fp.name} (Sa rows {Sa.shape[0]} != names {len(names)})")

        # GM 하나 = 샘플 하나 (feature 벡터 길이 nT)
        for i in range(Sa.shape[0]):
            f = Sa[i, :].astype(np.float32, copy=False)
            if f.shape[0] != nT_ref:
                raise ValueError(f"Sa row length mismatch: {fp.name} / i={i}")
            if not np.isfinite(f).all():
                raise ValueError(f"NaN/inf in {fp.name} / gm={names[i]}")

            cname = f"{fp.stem}__PSA__{str(names[i])}"
            X_cols.append(f)
            col_names.append(cname)
            y_rows.append([y1, y2, y3, y4])

    if not X_cols:
        raise RuntimeError("No usable samples found in npz files")

    X_arr = np.stack(X_cols, axis=0)  # (nSamples, nT)
    X_df = pd.DataFrame(X_arr.T, columns=col_names)
    y_df = pd.DataFrame(y_rows, columns=["y1", "y2", "y3", "y4"])

    X_df.to_csv(out_x, index=False)
    y_df.to_csv(out_y, index=False)

    return {"X_shape": list(X_df.shape), "y_shape": list(y_df.shape), "out_x": str(out_x), "out_y": str(out_y)}


def build_dataset(cfg: Dict[str, Any], input_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """
    input_format에 따라 xlsx 또는 npz를 읽어 dataset 생성.
      - cfg["input_format"] = "xlsx" | "npz"
    """
    fmt = str(cfg.get("input_format", "xlsx")).lower().strip()
    if fmt == "xlsx":
        return build_dataset_from_xlsx_folder(cfg, input_dir, out_dir)
    if fmt == "npz":
        return build_dataset_from_npz_folder(cfg, input_dir, out_dir)
    raise ValueError(f"Unknown input_format: {fmt} (use xlsx|npz)")