from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from src.lib.io import safe_filename


def split_target_xlsx_to_txt(cfg: Dict[str, Any], excel_path: Path, out_dir: Path) -> Dict[str, Any]:
    nd = int(cfg.get("filename_digits", 6))
    save_fmt = str(cfg.get("save_fmt", "%.10g"))
    delimiter = str(cfg.get("delimiter", "\t"))

    out_dir.mkdir(parents=True, exist_ok=True)

    def fmt_num(v, ndigits=nd):
        return f"{float(v):.{ndigits}g}"

    df_params = pd.read_excel(excel_path, sheet_name="params", engine="openpyxl")
    df_xy = pd.read_excel(excel_path, sheet_name="curve_xy", engine="openpyxl")

    written: List[Path] = []
    for _, row in df_params.iterrows():
        cid = row["curve_id"]
        colx = f"{cid}_x"
        coly = f"{cid}_y"
        if colx not in df_xy.columns or coly not in df_xy.columns:
            continue

        x = df_xy[colx].to_numpy(dtype=float)
        y = df_xy[coly].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]

        fname = f"{fmt_num(row['b'])}l{fmt_num(row['y_b'])}l{fmt_num(row['c'])}l{fmt_num(row['y_c'])}.txt"
        fname = safe_filename(fname)
        fpath = out_dir / fname
        np.savetxt(fpath, np.column_stack([x, y]), fmt=save_fmt, delimiter=delimiter)
        written.append(fpath)

    return {"num_txt": len(written), "out_dir": str(out_dir)}
