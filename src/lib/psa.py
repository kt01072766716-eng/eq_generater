from __future__ import annotations

import numpy as np
from math import pi


def compute_psa(acc, dt: float, T_array, zeta: float = 0.05):
    """Newmark-β (γ=1/2, β=1/4) pseudo spectral acceleration."""
    acc = np.asarray(acc, float)
    N = len(acc)
    psa = np.zeros(len(T_array), float)

    beta = 1.0 / 4.0
    gamma = 1.0 / 2.0

    for j, T in enumerate(T_array):
        w = 2.0 * pi / T
        k = w * w
        c = 2.0 * zeta * w

        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)

        k_hat = k + a0 + a1 * c

        u = v = a = 0.0
        umax = 0.0
        p = -acc

        for i in range(N - 1):
            p_n1 = p[i + 1]
            p_hat = (
                p_n1
                + a0 * u
                + a2 * v
                + a3 * a
                + c * (a1 * u + a4 * v + a5 * a)
            )

            u_new = p_hat / k_hat
            a_new = a0 * (u_new - u) - a2 * v - a3 * a
            v_new = v + dt * ((1.0 - gamma) * a + gamma * a_new)

            u, v, a = u_new, v_new, a_new
            if abs(u) > umax:
                umax = abs(u)

        psa[j] = umax * (2.0 * pi / T) ** 2

    return psa
