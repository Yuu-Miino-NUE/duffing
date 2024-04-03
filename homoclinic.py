import sys, json
from collections.abc import Callable
import numpy as np
from scipy.optimize import root

from duffing import poincare_map, Parameter
from fix import fix


class HomoclinicResult:
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
    ) -> None:
        self.success = success
        self.xu = xu
        self.xs = xs
        self.xh = xh
        self.xh_err = xh_err
        self.message = message

    def __repr__(self) -> str:
        return f"HomoclinicResult({self.success=}, {self.message=}, {self.xu=}, {self.xs=}, {self.xh=}, {self.xh_err=})"


def homoclinic_func(
    vars: np.ndarray,
    pmap_u: Callable[[np.ndarray], np.ndarray],
    pmap_s: Callable[[np.ndarray], np.ndarray],
    xfix: np.ndarray,
    norm_u: np.ndarray,
    norm_s: np.ndarray,
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

    Raises
    ------
    ValueError
        Invalid dimension of eigenspace. This error is raised when the dimension of the eigenspace is not 1.
    """

    fix_result = fix(xfix0, param, period)
    if not fix_result.success:
        print(fix_result.message)
        return

    x_fix = fix_result.x

    if fix_result.u_edim != 1 or fix_result.s_edim != 1:
        raise ValueError("Invalid dimension of eigenspace")

    unstable_vec = fix_result.u_evec[:, 0]
    stable_vec = fix_result.s_evec[:, 0]

    norm_mat = np.array([[0, 1], [-1, 0]])
    norm_u = norm_mat @ unstable_vec
    norm_s = norm_mat @ stable_vec

    u_itr_cnt = 2 if np.sign(fix_result.u_eig[0]) == -1 else 1
    s_itr_cnt = 2 if np.sign(fix_result.s_eig[0]) == -1 else 1

    unstable_func = lambda x: poincare_map(x, param, itr_cnt=u_itr_cnt * maps_u).x
    stable_func = lambda x: poincare_map(
        x, param, itr_cnt=s_itr_cnt * maps_s, inverse=True
    ).x

    func = lambda x: homoclinic_func(
        x, unstable_func, stable_func, x_fix, norm_u, norm_s
    )

    sol = root(func, np.concatenate((xu0, xs0)))

    if sol.success:
        xu = sol.x[0:2]
        xs = sol.x[2:4]
        xh_u = unstable_func(xu)
        xh_s = stable_func(xs)
        xh_err = xh_u - xh_s

        return HomoclinicResult(
            success=True,
            xu=xu,
            xs=xs,
            xh=xh_u,
            xh_err=xh_err,
            message="Success",
        )

    else:
        return HomoclinicResult(success=False, message=sol.message)


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

    print(homoclinic(x0, period, param, xu0, xs0, maps_u, maps_s))


if __name__ == "__main__":
    main()
