import sys, json
from collections.abc import Callable
import numpy as np
from scipy.optimize import root

from duffing import Parameter, PoincareMapResult, poincare_map
from fix import fix, fix_func
from homoclinic import homoclinic, homoclinic_func


def hbf_func(
    vars: np.ndarray,
    period: int,
    pmap: Callable[[np.ndarray, Parameter], PoincareMapResult],
    pmap_u: Callable[[np.ndarray, Parameter], PoincareMapResult],
    pmap_s: Callable[[np.ndarray, Parameter], PoincareMapResult],
    param: Parameter,
    param_key: str,
) -> np.ndarray:
    x_fix0 = vars[0:2]
    x_u0 = vars[2:4]
    x_s0 = vars[4:6]
    setattr(param, param_key, vars[6])

    fix_result = fix(x_fix0, param, period)
    if not fix_result.success:
        raise ValueError(fix_result.message)

    tvec_u = pmap_u(x_u0, param).jac @ fix_result.u_evec[:, 0]
    tvec_s = pmap_s(x_s0, param).jac @ fix_result.s_evec[:, 0]

    tvec_u /= np.linalg.norm(tvec_u)
    tvec_s /= np.linalg.norm(tvec_s)
    print(tvec_u, tvec_s)

    norm_mat = np.array([[0, 1], [-1, 0]])
    nvec_u = norm_mat @ fix_result.u_evec[:, 0]
    nvec_s = norm_mat @ fix_result.s_evec[:, 0]

    _pmap = lambda x: pmap(x, param).x
    _pmap_u = lambda x: pmap_u(x, param).x
    _pmap_s = lambda x: pmap_s(x, param).x

    ret = np.empty(7)
    ret[0:2] = fix_func(x_fix0, _pmap)
    ret[2:6] = homoclinic_func(
        vars[2:6], _pmap_u, _pmap_s, fix_result.x, nvec_u, nvec_s
    )
    ret[6] = np.linalg.det(np.vstack((tvec_u, tvec_s)).T)

    print(np.linalg.norm(ret[0:6]), ret[6])
    return ret


def hbf(
    xfix0: np.ndarray,
    period: int,
    param: Parameter,
    param_key: str,
    xu0: np.ndarray,
    xs0: np.ndarray,
    maps_u: int,
    maps_s: int,
):
    fix_result = fix(xfix0, param, period)
    homo_result = homoclinic(xfix0, period, param, xu0, xs0, maps_u, maps_s)

    if not fix_result.success or not homo_result.success:
        raise ValueError(fix_result.message)

    u_itr_cnt = 2 if np.sign(fix_result.u_eig[0]) == -1 else 1
    s_itr_cnt = 2 if np.sign(fix_result.s_eig[0]) == -1 else 1

    pmap = lambda x, p: poincare_map(x, p, calc_jac=True)
    pmap_u = lambda x, p: poincare_map(x, p, itr_cnt=u_itr_cnt * maps_u, calc_jac=True)
    pmap_s = lambda x, p: poincare_map(
        x, p, itr_cnt=s_itr_cnt * maps_s, calc_jac=True, inverse=True
    )

    func = lambda x: hbf_func(x, period, pmap, pmap_u, pmap_s, param, param_key)

    vars = np.empty(7)
    vars[0:6] = np.concatenate((xfix0, xu0, xs0))
    vars[6] = getattr(param, param_key)

    sol = root(func, vars)

    if sol.success:
        return sol.x


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

    print(hbf(x0, period, param, "B", xu0, xs0, maps_u, maps_s).tolist())


if __name__ == "__main__":
    main()
