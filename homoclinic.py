import sys, json
from collections.abc import Callable
import numpy as np
from scipy.optimize import root

from duffing import poincare_map, Parameter
from fix import fix


def homoclinic_func(
    vars: np.ndarray,
    pmap_u: Callable[[np.ndarray], np.ndarray],
    pmap_s: Callable[[np.ndarray], np.ndarray],
    xfix: np.ndarray,
    norm_u: np.ndarray,
    norm_s: np.ndarray,
) -> np.ndarray:
    xu = vars[0:2]
    xs = vars[2:4]

    ret = np.zeros(4)
    ret[0:2] = pmap_u(xu) - pmap_s(xs)
    ret[2] = np.dot(norm_u, xu - xfix)
    ret[3] = np.dot(norm_s, xs - xfix)

    return ret


def homoclinic(
    xfix0: np.ndarray,
    period: int,
    param: Parameter,
    xu0: np.ndarray,
    xs0: np.ndarray,
    maps_u: int,
    maps_s: int,
):

    fix_result = fix(xfix0, param, period)

    x_fix = fix_result["x"]

    if fix_result["u_edim"] != 1 or fix_result["s_edim"] != 1:
        raise ValueError("Invalid dimension of eigenspace")

    norm_mat = np.array([[0, 1], [-1, 0]])
    unstable_vec = fix_result["u_evec"][:, 0]
    norm_u = norm_mat @ unstable_vec
    stable_vec = fix_result["s_evec"][:, 0]
    norm_s = norm_mat @ stable_vec

    u_itr_cnt = 2 if np.sign(fix_result["u_eig"][0]) == -1 else 1
    s_itr_cnt = 2 if np.sign(fix_result["s_eig"][0]) == -1 else 1

    unstable_func = lambda x: poincare_map(x, param, itr_cnt=u_itr_cnt * maps_u)["x"]
    stable_func = lambda x: poincare_map(
        x, param, itr_cnt=s_itr_cnt * maps_s, inverse=True
    )["x"]

    func = lambda x: homoclinic_func(
        x, unstable_func, stable_func, x_fix, norm_u, norm_s
    )

    sol = root(func, np.concatenate((xu0, xs0)))

    if sol.success:
        print(sol.x.tolist())
    else:
        print(sol.message)


def main():
    try:
        with open(sys.argv[1], "r") as f:
            data = json.load(f)
        x0 = np.array(data.get("x0", [0, 0]))
        param = Parameter(**data.get("parameters", {}))
        period = data.get("period", 1)
        xu0 = np.array(data.get("xu", [0, 0]))
        xs0 = np.array(data.get("xs", [0, 0]))
        maps_u = data.get("maps_u", 1)
        maps_s = data.get("maps_s", 1)
    except IndexError:
        raise IndexError("Usage: python fix.py [data.json]")
    except FileNotFoundError:
        raise FileNotFoundError(f"{sys.argv[1]} not found")

    homoclinic(x0, period, param, xu0, xs0, maps_u, maps_s)


if __name__ == "__main__":
    main()
