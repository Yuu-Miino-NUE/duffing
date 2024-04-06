"""Module for calculating the homoclinic point of a fixed point."""

import sys, json
from collections.abc import Callable
import numpy
import numpy as np
from scipy.optimize import root

from core import IterItems
from manifold import prepare_by_fix
from system import poincare_map, Parameter


def calc_tvec_diff(
    jac_u: numpy.ndarray,
    jac_s: numpy.ndarray,
    eig_u: numpy.ndarray,
    eig_s: numpy.ndarray,
) -> float:
    """Calculate the difference of the tangent vectors at the homoclinic point.

    Parameters
    ----------
    jac_u : numpy.ndarray
        Jacobian matrix at the homoclinic point in the unstable manifold.
    jac_s : numpy.ndarray
        Jacobian matrix at the homoclinic point in the stable manifold.
    eig_u : numpy.ndarray
        Unstable eigenvector of the fixed point.
    eig_s : numpy.ndarray
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
    message : str
        Message of the calculation result.
    xu : numpy.ndarray
        Calculated homoclinic point in unstable eigenspace.
    xs : numpy.ndarray
        Calculated homoclinic point in stable eigenspace.
    xh : numpy.ndarray
        Calculated homoclinic point forward from xu and backward from xs.
    xh_err : float
        Error of the homoclinic point, calculated as xh_u - xh_s.
    tvec_diff: float
        Difference of the tangent vectors at the homoclinic point.
    xfix: numpy.ndarray
        Given fixed or periodic point.
    period : int
        Period of the periodic point, which is 1 for a fixed point.
    maps_u : int
        Given count of forward mapping from xu to xh.
    maps_s : int
        Given count of backward mapping from xs to xh.
    parameters : Parameter
        Given parameter object.
    """

    def __init__(
        self,
        success: bool,
        message: str,
        xu: numpy.ndarray | None = None,
        xs: numpy.ndarray | None = None,
        xh: numpy.ndarray | None = None,
        xh_err: numpy.ndarray | None = None,
        tvec_diff: float | None = None,
        xfix: numpy.ndarray | None = None,
        period: int | None = None,
        maps_u: int | None = None,
        maps_s: int | None = None,
        parameters: Parameter | None = None,
    ) -> None:
        self.success = success
        self.message = message
        self.xu = xu
        self.xs = xs
        self.xh = xh
        self.xh_err = xh_err
        self.tvec_diff = tvec_diff
        self.xfix = xfix
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
        return f"HomoclinicResult(\n{'\n'.join(self.out_strs())}\n"


def homoclinic_func(
    vars: numpy.ndarray,
    pmap_u: Callable[[numpy.ndarray], numpy.ndarray],
    pmap_s: Callable[[numpy.ndarray], numpy.ndarray],
    xfix: numpy.ndarray,
    norm_u: numpy.ndarray,
    norm_s: numpy.ndarray,
    allowed_dist: float = 1e-2,
) -> numpy.ndarray:
    """Function for evaluating homoclinic point.

    Parameters
    ----------
    vars : numpy.ndarray
        Initial points of the homoclinic points [xu, yu] and [xs, ys], which are enough close to the fixed point. They should form a 1-D array with the following order: [xu, yu, xs, ys].
    pmap_u : Callable[[numpy.ndarray], numpy.ndarray]
        Function to map from [xu, yu] to [xh, yh].
    pmap_s : Callable[[numpy.ndarray], numpy.ndarray]
        Function to map from [xs, ys] to [xh, yh].
    xfix : numpy.ndarray
        Fixed point.
    norm_u : numpy.ndarray
        Normal vector of unstable eigenvector.
    norm_s : numpy.ndarray
        Normal vector of stable eigenvector.
    allowed_dist : float, optional
        Allowed distance from the fixed point, by default 1e-2.

    Returns
    -------
    numpy.ndarray
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
    xfix0: numpy.ndarray,
    period: int,
    param: Parameter,
    xu0: numpy.ndarray,
    xs0: numpy.ndarray,
    maps_u: int,
    maps_s: int,
):
    """Homoclinic point calculation.

    Parameters
    ----------
    xfix0 : numpy.ndarray
        Initial value for the fixed or periodic point.
    period : int
        Period of the periodic point.
    param : Parameter
        Parameter object.
    xu0 : numpy.ndarray
        Initial point of the homoclinic point in the unstable manifold.
    xs0 : numpy.ndarray
        Initial point of the homoclinic point in the stable manifold.
    maps_u : int
        Count of forward mapping from xu0 to xh.
    maps_s : int
        Count of backward mapping from xs0 to xh.

    Returns
    -------
    HomoclinicResult
        Homoclinic point calculation result.

    """

    # Prepare the fixed point and eigenvectors
    xfix, u_evec, s_evec, u_itr_cnt, s_itr_cnt = prepare_by_fix(xfix0, param, period)

    # Normal vectors of the eigenvectors
    norm_mat = np.array([[0, 1], [-1, 0]])
    norm_u = norm_mat @ u_evec
    norm_s = norm_mat @ s_evec

    # Functions from xu (and xs) to xh
    u_func = lambda x: poincare_map(
        x, param, itr_cnt=u_itr_cnt * period * maps_u, calc_jac=True
    )
    s_func = lambda x: poincare_map(
        x, param, itr_cnt=s_itr_cnt * period * maps_s, inverse=True, calc_jac=True
    )

    pmap_u = lambda y: u_func(y).x
    pmap_s = lambda y: s_func(y).x

    # Replace to conceil unchanged parameters
    func = lambda x: homoclinic_func(
        vars=x,
        pmap_u=pmap_u,
        pmap_s=pmap_s,
        xfix=xfix,
        norm_u=norm_u,
        norm_s=norm_s,
    )

    # Main calculation
    sol = root(func, np.concatenate((xu0, xs0)))

    # Return the result
    if sol.success:
        xu = sol.x[0:2]
        xs = sol.x[2:4]
        xh_u = u_func(xu)
        xh_s = s_func(xs)
        xh_err = xh_u.x - xh_s.x

        if (jac_hu := xh_u.jac) is None:
            raise ValueError(
                "Failed to find Jacobian matrix at the homoclinic point in the unstable manifold"
            )
        if (jac_hs := xh_s.jac) is None:
            raise ValueError(
                "Failed to find Jacobian matrix at the homoclinic point in the stable manifold"
            )

        return HomoclinicResult(
            success=True,
            message="Success",
            xu=xu,
            xs=xs,
            xh=xh_u.x,
            xh_err=xh_err,
            xfix=xfix,
            tvec_diff=calc_tvec_diff(jac_hu, jac_hs, u_evec, s_evec),
            period=period,
            maps_u=maps_u,
            maps_s=maps_s,
            parameters=param,
        )

    else:
        return HomoclinicResult(success=False, message=sol.message)


def _main():
    # Load from JSON file
    try:
        with open(sys.argv[1], "r") as f:
            data = json.load(f)
        xfix0 = np.array(data.get("xfix", [0, 0]))
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

    # Calculate the homoclinic point
    res = homoclinic(xfix0, period, param, xu0, xs0, maps_u, maps_s)

    # Print the result
    print(dict(res))
    res.dump(sys.stdout)


if __name__ == "__main__":
    _main()
