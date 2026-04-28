from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


def _logspace_or_single(vmin: float, vmax: float, n: int, v0: float) -> np.ndarray:
    if n == 1:
        return np.array([v0], dtype=float)
    return np.logspace(np.log10(vmin), np.log10(vmax), n)


def _split_half(n: int) -> tuple[int, int]:
    left = n // 2
    right = n - left
    return left, right


def generate_target_spectra_xlsx(cfg: Dict[str, Any], out_xlsx: Path, make_plots: bool = False) -> Dict[str, Any]:
    """Generate curves and save to an Excel (curve_xy, params, skipped). Returns a manifest dict."""

    x_min_fixed = float(cfg["x_min_fixed"])
    x_max_fixed = float(cfg["x_max_fixed"])

    a0 = float(cfg["a0"])
    b0 = float(cfg["b0"])
    c0 = float(cfg["c0"])
    d0 = float(cfg["d0"])
    y_b = float(cfg["y_b"])

    # Freedom 1: b sweep (a -> b)
    b_min = float(cfg["b_min"])
    b_max = float(cfg["b_max"])
    Nb = int(cfg["Nb"])

    # Freedom 2: c0 sweep in base coordinates, actual c = c0v * s
    c0_min = float(cfg["c0_min"])
    c0_max = float(cfg["c0_max"])
    Nc = int(cfg["Nc"])

    # c-b gap constraints configurable in YAML
    min_cb_gap = float(cfg["min_cb_gap"])
    max_cb_gap = float(cfg["max_cb_gap"])

    r_pos_max = float(cfg["r_pos_max"])
    r_neg_min = float(cfg["r_neg_min"])
    N_pos = int(cfg["N_pos"])
    N_neg = int(cfg["N_neg"])
    include_zero = bool(cfg["include_zero"])

    N1 = int(cfg["N1"])
    N2 = int(cfg["N2"])
    N3 = int(cfg["N3"])
    N4 = int(cfg["N4"])
    N5 = int(cfg["N5"])
    TOTAL_POINTS = N1 + N2 + N3 + N4 + N5

    b_values = _logspace_or_single(b_min, b_max, Nb, b0)
    c0_values = _logspace_or_single(c0_min, c0_max, Nc, c0)

    def make_thetas_for_bc(b: float, c: float):
        L = np.log(c / b)
        if L <= 0:
            raise ValueError(f"Need c>b. Got c={c}, b={b}")

        tan_th_pos_max = (r_pos_max - 1.0) * y_b / L
        theta_pos_max = np.degrees(np.arctan(tan_th_pos_max))

        tan_th_neg_min = (r_neg_min - 1.0) * y_b / L
        theta_neg_min = np.degrees(np.arctan(tan_th_neg_min))

        thetas: List[float] = []
        if N_neg > 0:
            thetas += list(np.linspace(theta_neg_min, 0.0, N_neg, endpoint=False))
        if include_zero:
            thetas += [0.0]
        if N_pos > 0:
            thetas += list(np.linspace(0.0, theta_pos_max, N_pos + 1)[1:])

        return thetas, float(theta_neg_min), float(theta_pos_max)

    def build_curve(a: float, b: float, c: float, d: float, theta_deg: float):
        cb_gap = c - b
        if cb_gap < min_cb_gap:
            raise ValueError(f"Skip curve because c-b={cb_gap:.6f} < {min_cb_gap}")
        if cb_gap > max_cb_gap:
            raise ValueError(f"Skip curve because c-b={cb_gap:.6f} > {max_cb_gap}")
        if not (b < c):
            raise ValueError(f"Need b<c. Got b={b}, c={c}")

        theta = np.deg2rad(theta_deg)
        alpha = np.tan(theta)
        beta = y_b - alpha * np.log(b)

        def y3(x):
            return alpha * np.log(x) + beta

        y_c = float(y3(c))
        ratio = y_c / y_b

        y0 = (3 * y_c + 2 * y_b) / 5 / 2.5
        if y_b > y_c:
            y0 = (2 * y_c + 3 * y_b) / 5 / 2.5

        m2 = (y_b - y0) / (b - a)

        def y2(x):
            return y0 + m2 * (x - a)

        k1 = y_c * c
        k2 = k1 * d

        # truncated segments -> redistribute removed nodes to seg2/seg3
        seg1_cut = a <= x_min_fixed
        seg5_cut = d >= x_max_fixed

        add2 = 0
        add3 = 0
        if seg1_cut:
            t2, t3 = _split_half(N1)
            add2 += t2
            add3 += t3
        if seg5_cut:
            t2, t3 = _split_half(N5)
            add2 += t2
            add3 += t3

        n1_use = 0 if seg1_cut else N1
        n2_use = N2 + add2
        n3_use = N3 + add3
        n4_use = N4
        n5_use = 0 if seg5_cut else N5

        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        if not seg1_cut:
            x1 = np.logspace(np.log10(x_min_fixed), np.log10(a), n1_use)
            x1[0] = x_min_fixed
            x1[-1] = a
            y1 = np.full_like(x1, y0)

            x2 = np.linspace(a, b, n2_use)
            y2v = y2(x2)

            x_parts.append(x1)
            y_parts.append(y1)
            x_parts.append(x2)
            y_parts.append(y2v)
        else:
            x2 = np.linspace(x_min_fixed, b, n2_use)
            y2v = y2(x2)
            x_parts.append(x2)
            y_parts.append(y2v)

        x3 = np.linspace(b, c, n3_use)
        y3v = y3(x3)
        x_parts.append(x3)
        y_parts.append(y3v)

        d_eff = min(d, x_max_fixed)
        if d_eff > c:
            x4 = np.linspace(c, d_eff, n4_use)
            x4[-1] = d_eff
            y4v = k1 / x4
            x_parts.append(x4)
            y_parts.append(y4v)

        if not seg5_cut:
            x5 = np.logspace(np.log10(d), np.log10(x_max_fixed), n5_use)
            x5[0] = d
            x5[-1] = x_max_fixed
            y5v = k2 / (x5 ** 2)
            x_parts.append(x5)
            y_parts.append(y5v)

        x = np.concatenate(x_parts)
        y = np.concatenate(y_parts)

        if abs(x[-1] - x_max_fixed) > 1e-9:
            raise RuntimeError(
                f"[BUG] x_end={x[-1]} != x_max_fixed={x_max_fixed} "
                f"(a={a}, b={b}, c={c}, d={d}, d_eff={d_eff})"
            )

        if len(x) != TOTAL_POINTS:
            raise RuntimeError(
                f"[BUG] len(x)={len(x)} != TOTAL_POINTS={TOTAL_POINTS} "
                f"(a={a}, b={b}, c={c}, d={d}, n1={n1_use}, n2={n2_use}, n3={n3_use}, n4={n4_use}, n5={n5_use})"
            )

        info = {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "d": float(d),
            "d_eff": float(d_eff),
            "theta_deg": float(theta_deg),
            "alpha": float(alpha),
            "beta": float(beta),
            "y_b": float(y_b),
            "y_c": float(y_c),
            "ratio_yc_yb": float(ratio),
            "y0": float(y0),
            "m2": float(m2),
            "k1": float(k1),
            "k2": float(k2),
            "cb_gap": float(cb_gap),
            "seg1_cut": bool(seg1_cut),
            "seg5_cut": bool(seg5_cut),
            "n1_used": int(n1_use),
            "n2_used": int(n2_use),
            "n3_used": int(n3_use),
            "n4_used": int(n4_use),
            "n5_used": int(n5_use),
            "n_total": int(len(x)),
        }
        return x, y, info

    xy_dict: Dict[str, pd.Series] = {}
    params_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    curve_idx = 0

    for ib, b in enumerate(b_values, start=1):
        # reverse of original scaling: choose b, then move a/c/d with same scale
        s = b / b0
        a = a0 * s
        d = d0 * s

        for ic, c0v in enumerate(c0_values, start=1):
            c = c0v * s
            cb_gap = c - b

            if cb_gap < min_cb_gap:
                skipped_rows.append(
                    {
                        "b_idx": ib,
                        "c_idx": ic,
                        "a": float(a),
                        "b": float(b),
                        "c": float(c),
                        "d": float(d),
                        "scale_s": float(s),
                        "cb_gap": float(cb_gap),
                        "reason": f"c-b={cb_gap:.6f} < {min_cb_gap}",
                    }
                )
                continue

            if cb_gap > max_cb_gap:
                skipped_rows.append(
                    {
                        "b_idx": ib,
                        "c_idx": ic,
                        "a": float(a),
                        "b": float(b),
                        "c": float(c),
                        "d": float(d),
                        "scale_s": float(s),
                        "cb_gap": float(cb_gap),
                        "reason": f"c-b={cb_gap:.6f} > {max_cb_gap}",
                    }
                )
                continue

            try:
                thetas, th_neg_min, th_pos_max = make_thetas_for_bc(b, c)
            except ValueError as e:
                skipped_rows.append(
                    {
                        "b_idx": ib,
                        "c_idx": ic,
                        "a": float(a),
                        "b": float(b),
                        "c": float(c),
                        "d": float(d),
                        "scale_s": float(s),
                        "cb_gap": float(cb_gap),
                        "reason": str(e),
                    }
                )
                continue

            for it, th in enumerate(thetas, start=1):
                try:
                    x, y, info = build_curve(a, b, c, d, th)
                except ValueError as e:
                    skipped_rows.append(
                        {
                            "b_idx": ib,
                            "c_idx": ic,
                            "theta_idx": it,
                            "a": float(a),
                            "b": float(b),
                            "c": float(c),
                            "d": float(d),
                            "theta_deg": float(th),
                            "scale_s": float(s),
                            "cb_gap": float(cb_gap),
                            "reason": str(e),
                        }
                    )
                    continue

                curve_idx += 1
                curve_id = f"curve_{curve_idx:03d}"
                info.update(
                    {
                        "curve_id": curve_id,
                        "b_idx": ib,
                        "c_idx": ic,
                        "theta_idx": it,
                        "scale_s": float(s),
                        "theta_neg_min_deg": float(th_neg_min),
                        "theta_pos_max_deg": float(th_pos_max),
                    }
                )
                params_rows.append(info)
                xy_dict[f"{curve_id}_x"] = pd.Series(x)
                xy_dict[f"{curve_id}_y"] = pd.Series(y)

    df_xy = pd.DataFrame(xy_dict)
    df_params = pd.DataFrame(params_rows)
    df_skipped = pd.DataFrame(skipped_rows)

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_xy.to_excel(writer, sheet_name="curve_xy", index=False)
        df_params.to_excel(writer, sheet_name="params", index=False)
        df_skipped.to_excel(writer, sheet_name="skipped", index=False)

    if make_plots:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 6))
        for i in range(curve_idx):
            cid = f"curve_{i+1:03d}"
            xvals = df_xy[f"{cid}_x"].dropna().to_numpy()
            yvals = df_xy[f"{cid}_y"].dropna().to_numpy()
            plt.plot(xvals, yvals, linewidth=1.1)
        plt.xscale("log")
        plt.xlim(x_min_fixed, x_max_fixed)
        plt.grid(True, which="both")
        plt.xlabel("x (log scale)")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()

    return {
        "num_curves": int(curve_idx),
        "num_skipped": int(len(skipped_rows)),
        "out_xlsx": str(out_xlsx),
        "sheets": ["curve_xy", "params", "skipped"],
    }
