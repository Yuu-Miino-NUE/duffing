import sys, json
from typing import Callable
import numpy as np
from scipy.optimize import root

from duffing import poincare_map, Parameter


def fix_func(vec_x: np.ndarray, pmap: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return pmap(vec_x) - vec_x


def fix(
    vec_x: np.ndarray, param: Parameter, period: int, verbose: bool = False
) -> dict[str, np.ndarray | str | bool]:
    x0 = vec_x.copy()
    pmap = lambda x: poincare_map(x, param, itr_cnt=period)["x"]

    sol = root(fix_func, x0, args=(pmap,))

    if sol.success:
        xfix = sol.x
        jac = poincare_map(xfix, param, itr_cnt=period, calc_jac=True)["jac"]
        eig, vec = np.linalg.eig(jac)

        return {
            "success": True,
            "x": xfix,
            "eig": eig,
            "abs_eig": np.abs(eig),
            "vec": vec,
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
