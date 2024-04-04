import sys, json
from collections.abc import Callable
import numpy as np
from scipy.optimize import root

from core import IterItems
from duffing import poincare_map, Parameter
from fix import fix


def tvec_diff(
    jac_u: np.ndarray, jac_s: np.ndarray, eig_u: np.ndarray, eig_s: np.ndarray
) -> float:
    """Calculate the difference of the tangent vectors at the homoclinic point.

    Parameters
    ----------
    jac_u : np.ndarray
        Jacobian matrix at the homoclinic point in the unstable manifold.
    jac_s : np.ndarray
        Jacobian matrix at the homoclinic point in the stable manifold.
    eig_u : np.ndarray
        Unstable eigenvector of the fixed point.
    eig_s : np.ndarray
        Stable eigenvector of the fixed point.

    Returns
    -------
    float
        Difference of the tangent vectors at the homoclinic point.
    """
    tvec_u = jac_u @ eig_u
    tvec_s = jac_s @ eig_s

    tvec_u /= np.linalg.norm(tvec_u)
    tvec_s /= np.linalg.norm(tvec_s)

    return np.linalg.det(np.vstack((tvec_u, tvec_s)).T)


class HomoclinicResult(IterItems):
    """Homoclinic point calculation result.

    Attributes
    ----------
    success : bool
        Success flag.
    xu : np.ndarray
        Homoclinic point in unstable eigenspace.
    xs : np.ndarray
        Homoclinic point in stable eigenspace.
    xh : np.ndarray
        Homoclinic point forward from xu and backward from xs.
    xh_err : float
        Error of the homoclinic point, calculated as xh_u - xh_s.
    xfix: np.ndarray
        Fixed point.
    tvec_diff: float
        Difference of the tangent vectors at the homoclinic point.
    message : str
        Message of the calculation result.
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
        period: int = 1,
        maps_u: int = -1,
        maps_s: int = -1,
        parameters: Parameter = Parameter(),
        tvec_diff: float = 0,
    ) -> None:
        self.success = success
        self.message = message
        self.xu = xu
        self.xs = xs
        self.xh = xh
        self.xh_err = xh_err
        self.xfix = xfix
        self.tvec_diff = tvec_diff
        self.period = period
        self.maps_u = maps_u
        self.maps_s = maps_s
        self.parameters = parameters

        super().__init__(
            [
                "xfix",
                "xu",
                "xs",
                "parameters",
                "period",
                "maps_u",
                "maps_s",
            ]
        )

    def __repr__(self) -> str:
        return f"HomoclinicResult({self.success=}, {self.message=}, {self.xu=}, {self.xs=}, {self.xh=}, {self.xh_err=}, {self.xfix=}, {self.tvec_diff=})"


def homoclinic_func(
    vars: np.ndarray,
    pmap_u: Callable[[np.ndarray], np.ndarray],
    pmap_s: Callable[[np.ndarray], np.ndarray],
    xfix: np.ndarray,
    norm_u: np.ndarray,
    norm_s: np.ndarray,
    allowed_dist: float = 1e-2,
) -> np.ndarray:
    """Function for evaluating homoclinic point.

    Parameters
    ----------
    vars : np.ndarray
        Initial points of the homoclinic points [xu, yu] and [xs, ys], which are enough close to the fixed point. They should form a 1-D array with the following order: [xu, yu, xs, ys].
    pmap_u : Callable[[np.ndarray], np.ndarray]
        Function to map from [xu, yu] to [xh, yh].
    pmap_s : Callable[[np.ndarray], np.ndarray]
        Function to map from [xs, ys] to [xh, yh].
    xfix : np.ndarray
        Fixed point.
    norm_u : np.ndarray
        Normal vector of unstable eigenvector.
    norm_s : np.ndarray
        Normal vector of stable eigenvector.

    Returns
    -------
    np.ndarray
        Residuals of the homoclinic points.
    """
    xu = vars[0:2]
    xs = vars[2:4]

    vu = xu - xfix
    vs = xs - xfix

    if (nu := np.linalg.norm(vu)) > allowed_dist:
        raise ValueError(f"Invalid xu: too far from the fixed point. {nu}")
    if (ns := np.linalg.norm(vs)) > allowed_dist:
        raise ValueError(f"Invalid xs: too far from the fixed point. {ns}")

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
    """Homoclinic point calculation.

    Parameters
    ----------
    xfix0 : np.ndarray

    period : int
        Period of the periodic point.
    param : Parameter
        Parameter object.
    xu0 : np.ndarray
        Initial point of the homoclinic point in the unstable manifold.
    xs0 : np.ndarray
        Initial point of the homoclinic point in the stable manifold.
    maps_u : int
        Count of forward mapping from xu0 to xh.
    maps_s : int
        Count of backward mapping from xs0 to xh.

    Returns
    -------
    HomoclinicResult
        Homoclinic point calculation result.

    Raises
    ------
    ValueError
        Invalid dimension of eigenspace. This error is raised when the dimension of the eigenspace is not 1.
    """

    fix_result = fix(xfix0, param, period)
    if not fix_result.success:
        raise ValueError(fix_result.message)

    x_fix = fix_result.xfix

    if fix_result.u_edim != 1 or fix_result.s_edim != 1:
        raise ValueError("Invalid dimension of eigenspace")

    unstable_vec = fix_result.u_evec[:, 0]
    stable_vec = fix_result.s_evec[:, 0]

    norm_mat = np.array([[0, 1], [-1, 0]])
    norm_u = norm_mat @ unstable_vec
    norm_s = norm_mat @ stable_vec

    u_itr_cnt = 2 if np.sign(fix_result.u_eig[0]) == -1 else 1
    s_itr_cnt = 2 if np.sign(fix_result.s_eig[0]) == -1 else 1

    unstable_func = lambda x: poincare_map(
        x, param, itr_cnt=u_itr_cnt * maps_u, calc_jac=True
    )
    stable_func = lambda x: poincare_map(
        x, param, itr_cnt=s_itr_cnt * maps_s, inverse=True, calc_jac=True
    )

    func = lambda x: homoclinic_func(
        x,
        lambda x: unstable_func(x).x,
        lambda x: stable_func(x).x,
        x_fix,
        norm_u,
        norm_s,
    )

    sol = root(func, np.concatenate((xu0, xs0)))

    if sol.success:
        xu = sol.x[0:2]
        xs = sol.x[2:4]
        xh_u = unstable_func(xu)
        xh_s = stable_func(xs)
        xh_err = xh_u.x - xh_s.x

        return HomoclinicResult(
            success=True,
            xu=xu,
            xs=xs,
            xh=xh_u.x,
            xh_err=xh_err,
            xfix=x_fix,
            tvec_diff=tvec_diff(xh_u.jac, xh_s.jac, unstable_vec, stable_vec),
            period=period,
            maps_u=maps_u,
            maps_s=maps_s,
            parameters=param,
            message="Success",
        )

    else:
        return HomoclinicResult(success=False, message=sol.message)


def _main():
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
    except IndexError:
        raise IndexError("Usage: python fix.py [data.json]")
    except FileNotFoundError:
        raise FileNotFoundError(f"{sys.argv[1]} not found")

    res = homoclinic(x0, period, param, xu0, xs0, maps_u, maps_s)
    print(dict(res))
    res.dump(sys.stdout)


if __name__ == "__main__":
    _main()
