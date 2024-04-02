import sys, json
from typing import Any
from collections.abc import Callable
import numpy as np
from scipy.optimize import root

from duffing import poincare_map, Parameter


def fix_func(vec_x: np.ndarray, pmap: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return pmap(vec_x) - vec_x


def fix(
    vec_x: np.ndarray, param: Parameter, period: int, verbose: bool = False
) -> dict[str, Any]:
    x0 = vec_x.copy()
    pmap = lambda x: poincare_map(x, param, itr_cnt=period)["x"]
    func = lambda x: fix_func(x, pmap)

    sol = root(func, x0)

    if sol.success:
        xfix = sol.x
        jac = poincare_map(xfix, param, itr_cnt=period, calc_jac=True)["jac"]
        eig, vec = np.linalg.eig(jac)

        u_edim = np.sum(np.abs(np.real(eig)) > 1)
        c_edim = np.sum(np.abs(np.real(eig)) == 1)
        s_edim = np.sum(np.abs(np.real(eig)) < 1)

        u_eigs = eig[np.abs(np.real(eig)) > 1]
        c_eigs = eig[np.abs(np.real(eig)) == 1]
        s_eigs = eig[np.abs(np.real(eig)) < 1]

        u_evecs = vec[:, np.abs(np.real(eig)) > 1]
        c_evecs = vec[:, np.abs(np.real(eig)) == 1]
        s_evecs = vec[:, np.abs(np.real(eig)) < 1]

        return {
            "success": True,
            "x": xfix,
            "eig": eig,
            "abs_eig": np.abs(eig),
            "evec": vec,
            "u_edim": u_edim,
            "c_edim": c_edim,
            "s_edim": s_edim,
            "u_eig": u_eigs,
            "c_eig": c_eigs,
            "s_eig": s_eigs,
            "u_evec": u_evecs,
            "c_evec": c_evecs,
            "s_evec": s_evecs,
        }
    else:
        return {"success": False, "error": sol.message}


def main():
    try:
        with open(sys.argv[1], "r") as f:
            data = json.load(f)
        x0 = np.array(data.get("x0", [0, 0]))
        param = Parameter(**data.get("parameters", {}))
        period = data.get("period", 1)
    except IndexError:
        raise IndexError("Usage: python fix.py [data.json]")
    except FileNotFoundError:
        raise FileNotFoundError(f"{sys.argv[1]} not found")

    print(fix(x0, param, period))


if __name__ == "__main__":
    main()