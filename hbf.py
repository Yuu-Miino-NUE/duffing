import sys, json
from collections.abc import Callable
import numpy as np
from scipy.optimize import root

from core import IterItems
from duffing import Parameter, PoincareMapResult, poincare_map
from fix import fix, fix_func
from homoclinic import homoclinic, homoclinic_func, tvec_diff


class HbfResult(IterItems):
    """Homoclinic bifurcation point calculation result.

    Attributes
    ----------
    success : bool
        Success flag.
    message : str
        Message of the calculation result.
    xu : np.ndarray
        Homoclinic point in the unstable manifold.
    xs : np.ndarray
        Homoclinic point in the stable manifold.
    xh : np.ndarray
        Homoclinic point.
    xh_err : np.ndarray
        Error of the homoclinic point, calculated as xh_u - xh_s.
    xfix : np.ndarray
        Fixed point or periodic point.
    tvec_diff : float
        Difference of the tangent vectors at the homoclinic point.
    parameters : Parameter
        Parameter object.
    maps_u : int
        Count of forward mapping from xu to xh.
    maps_s : int
        Count of backward mapping from xs to xh.
    hbf_param_key : str
        Bifurcation parameter key to control.
    period : int
        Period of the fixed or periodic point.
    """

    def __init__(
        self,
        success: bool,
        message: str,
        xu: np.ndarray = np.empty(2),
        xs: np.ndarray = np.empty(2),
        xh: np.ndarray = np.empty(2),
        xh_err: np.ndarray = np.empty(2),
        xfix: np.ndarray = np.empty(2),
        tvec_diff: float = 0,
        parameters: Parameter = Parameter(),
        maps_u: int = -1,
        maps_s: int = -1,
        hbf_param_key: str = "",
        period: int = 1,
    ) -> None:
        self.success = success
        self.message = message
        self.xu = xu
        self.xs = xs
        self.xh = xh
        self.xh_err = xh_err
        self.xfix = xfix
        self.tvec_diff = tvec_diff
        self.parameters = parameters
        self.maps_u = maps_u
        self.maps_s = maps_s
        self.hbf_param_key = hbf_param_key
        self.period = period
        super().__init__(
            [
                "xfix",
                "xu",
                "xs",
                "parameters",
                "period",
                "maps_u",
                "maps_s",
                "hbf_param_key",
            ]
        )

    def __repr__(self) -> str:
        return f"HbfResult({self.success}, {self.message}, {self.xu}, {self.xs}, {self.xh}, {self.xh_err}, {self.xfix}, {self.tvec_diff}, {self.parameters})"


def hbf_func(
    vars: np.ndarray,
    period: int,
    pmap: Callable[[np.ndarray, Parameter], PoincareMapResult],
    pmap_u: Callable[[np.ndarray, Parameter], PoincareMapResult],
    pmap_s: Callable[[np.ndarray, Parameter], PoincareMapResult],
    param: Parameter,
    param_key: str,
    verbose: bool = False,
) -> np.ndarray:
    """Function for evaluating homoclinic bifurcation point.

    Parameters
    ----------
    vars : np.ndarray
        Initial points of the fixed or periodic point, homoclinic point in the unstable manifold, homoclinic point in the stable manifold, and bifurcation parameter. The shape of the array is (7,).
    period : int
        Period of the periodic point. For fixed point, set 1.
    pmap : Callable[[np.ndarray, Parameter], PoincareMapResult]
        Poincare map function.
    pmap_u : Callable[[np.ndarray, Parameter], PoincareMapResult]
        Function to map from [xu, yu] to [xh, yh].
    pmap_s : Callable[[np.ndarray, Parameter], PoincareMapResult]
        Function to map from [xs, ys] to [xh, yh].
    param : Parameter
        Parameter object.
    param_key : str
        Bifurcation parameter key to control.
    verbose : bool, optional
        Print progress, by default False.

    Returns
    -------
    np.ndarray
        Residual vector.

    Raises
    ------
    ValueError
        If the fixed point calculation fails.
    """
    x_fix0 = vars[0:2]
    x_u0 = vars[2:4]
    x_s0 = vars[4:6]
    setattr(param, param_key, vars[6])

    fix_result = fix(x_fix0, param, period)
    if not fix_result.success:
        raise ValueError(fix_result.message)

    norm_mat = np.array([[0, 1], [-1, 0]])
    nvec_u = norm_mat @ fix_result.u_evec[:, 0]
    nvec_s = norm_mat @ fix_result.s_evec[:, 0]

    _pmap = lambda x: pmap(x, param).x
    _pmap_u = lambda x: pmap_u(x, param).x
    _pmap_s = lambda x: pmap_s(x, param).x

    ret = np.empty(7)
    ret[0:2] = fix_func(x_fix0, _pmap)
    ret[2:6] = homoclinic_func(
        vars[2:6], _pmap_u, _pmap_s, fix_result.xfix, nvec_u, nvec_s
    )
    ret[6] = tvec_diff(
        pmap_u(x_u0, param).jac,
        pmap_s(x_s0, param).jac,
        fix_result.u_evec[:, 0],
        fix_result.s_evec[:, 0],
    )

    if verbose:
        print("Running... :", [f"{r:+.3e}" for r in ret], end="\r")
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
    verbose: bool = False,
) -> HbfResult:
    """Homoclinic point calculation.

    Parameters
    ----------
    xfix0 : np.ndarray
        Fixed or periodic point.
    period : int
        Period of the fixed or periodic point.
    param : Parameter
        Parameter object.
    param_key : str
        Bifurcation parameter key to control.
    xu0 : np.ndarray
        Initial point of the homoclinic point in the unstable manifold.
    xs0 : np.ndarray
        Initial point of the homoclinic point in the stable manifold.
    maps_u : int
        Count of forward mapping from xu0 to xh.
    maps_s : int
        Count of backward mapping from xs0 to xh.
    verbose : bool, optional
        Print progress, by default False.

    Returns
    -------
    HbfResult
        Homoclinic point calculation result.

    Raises
    ------
    ValueError
        If the fixed point calculation fails.
    ValueError
        If the homoclinic point calculation fails.
    """
    fix_result = fix(xfix0, param, period)
    homo_result = homoclinic(xfix0, period, param, xu0, xs0, maps_u, maps_s)

    if not fix_result.success:
        raise ValueError(fix_result.message)
    if not homo_result.success:
        raise ValueError(homo_result.message)

    u_itr_cnt = 2 if np.sign(fix_result.u_eig[0]) == -1 else 1
    s_itr_cnt = 2 if np.sign(fix_result.s_eig[0]) == -1 else 1

    pmap = lambda x, p: poincare_map(x, p, calc_jac=True)
    pmap_u = lambda x, p: poincare_map(x, p, itr_cnt=u_itr_cnt * maps_u, calc_jac=True)
    pmap_s = lambda x, p: poincare_map(
        x, p, itr_cnt=s_itr_cnt * maps_s, calc_jac=True, inverse=True
    )

    func = lambda x: hbf_func(
        x, period, pmap, pmap_u, pmap_s, param, param_key, verbose
    )

    vars = np.empty(7)
    vars[0:6] = np.concatenate((xfix0, xu0, xs0))
    vars[6] = getattr(param, param_key)

    sol = root(func, vars)
    if verbose:
        print()

    if sol.success:
        xfix = sol.x[0:2]
        xu = sol.x[2:4]
        xs = sol.x[4:6]
        setattr(param, param_key, sol.x[6])

        fix_result = fix(xfix, param, period)
        xh_u = pmap_u(xu, param)
        xh_s = pmap_s(xs, param)
        xh_err = xh_u.x - xh_s.x

        return HbfResult(
            success=True,
            message="Success",
            xu=xu,
            xs=xs,
            xh=xh_u.x,
            xh_err=xh_err,
            xfix=xfix,
            tvec_diff=tvec_diff(
                xh_u.jac, xh_s.jac, fix_result.u_evec[:, 0], fix_result.s_evec[:, 0]
            ),
            parameters=param,
            maps_u=maps_u,
            maps_s=maps_s,
            hbf_param_key=param_key,
            period=period,
        )
    else:
        return HbfResult(success=False, message=sol.message)


def main():
    try:
        with open(sys.argv[1], "r") as f:
            data = json.load(f)
        x0 = np.array(data.get("xfix", [0, 0]))
        param = Parameter(**data.get("parameters", {}))
        period = data.get("period", 1)
        xu0 = np.array(data.get("xu", [0, 0]))
        xs0 = np.array(data.get("xs", [0, 0]))
        maps_u = data.get("maps_u", 1)
        maps_s = data.get("maps_s", 1)
        param_key = data.get("hbf_param_key", "B")
    except IndexError:
        raise IndexError("Usage: python fix.py [data.json]")
    except FileNotFoundError:
        raise FileNotFoundError(f"{sys.argv[1]} not found")

    res = hbf(x0, period, param, param_key, xu0, xs0, maps_u, maps_s, verbose=True)
    print(res)
    res.dump(sys.stdout)


if __name__ == "__main__":
    main()
