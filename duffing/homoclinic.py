"""Module for calculating the homoclinic point of a fixed point.

Examples
--------
Prepare a JSON file with the following content:

.. code-block:: json

    {
        "xfix": [-0.95432, 0.19090],
        "period": 1,
        "parameters": {
            "k": 0.05,
            "B0": 0,
            "B": 0.3
        },
        "xu": [-0.95430982, 0.19084544],
        "xs": [-0.95430534, 0.19092100],
        "maps_u": 7,
        "maps_s": 6
    }

Run the following code in your terminal with the JSON file:

.. code-block:: bash

    python homoclinic.py [data.json]

The result will be printed and dumped as a JSON file with the same name as the input file but with "_homoclinic" suffix.
The content of the dumped file will be like below:

.. code-block:: json

    {
        "xfix": [
            -0.9543211072836386,
            0.19089923612533358
        ],
        "xu": [
            -0.9543098289790618,
            0.1908454373599407
        ],
        "xs": [
            -0.954305345456587,
            0.19092100473461604
        ],
        "parameters": {
            "k": 0.05,
            "B": 0.3,
            "B0": 0
        },
        "period": 1,
        "maps_u": 7,
        "maps_s": 6
    }



"""

import sys, json
from collections.abc import Callable
import numpy
import numpy as np
from scipy.optimize import root

from core import IterItems
from manifold import _prepare_by_fix
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
        return f"HomoclinicResult(\n{'\n'.join(self.out_strs())}\n)"


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
    xfix: numpy.ndarray,
    period: int,
    param: Parameter,
    xu: numpy.ndarray,
    xs: numpy.ndarray,
    maps_u: int,
    maps_s: int,
):
    """Homoclinic point calculation.

    Parameters
    ----------
    xfix : numpy.ndarray
        Initial value for the fixed or periodic point.
    period : int
        Period of the periodic point.
    param : Parameter
        Parameter object.
    xu : numpy.ndarray
        Initial point of the homoclinic point in the unstable manifold.
    xs : numpy.ndarray
        Initial point of the homoclinic point in the stable manifold.
    maps_u : int
        Count of forward mapping from xu0 to xh.
    maps_s : int
        Count of backward mapping from xs0 to xh.

    Returns
    -------
    HomoclinicResult
        Homoclinic point calculation result.

    Example
    -------

    .. code-block:: python

        import numpy as np
        from system import Parameter
        from homoclinic import homoclinic

        homoclinic_args = {
            "xfix": np.array([[-0.95432, 0.19090]]),
            "period": 1,
            "param": Parameter(k=0.05, B0=0, B=0.3),
            "xu": np.array([-0.95430982, 0.19084544]),
            "xs": np.array([-0.95430534, 0.19092100]),
            "maps_u": 7,
            "maps_s": 6,
        }

        res = homoclinic(**homoclinic_args)
        print(res)

    The above code will print the homoclinic point calculation result like below:

    .. code-block:: python

        HomoclinicResult(
            success: True
            message: Success
            xu: [-0.95430983  0.19084544]
            xs: [-0.95430535  0.190921  ]
            xh: [0.13204358 0.06471627]
            xh_err: [ 1.51237094e-08 -6.84885871e-08]
            tvec_diff: 0.9930997598141681
            xfix: [-0.95432111  0.19089924]
            period: 1
            maps_u: 7
            maps_s: 6
            parameters: (k, B0, B) = (+0.050000, +0.000000, +0.300000)
        )

    """

    # Prepare the fixed point and eigenvectors
    _xfix, u_evec, s_evec, u_itr_cnt, s_itr_cnt = _prepare_by_fix(xfix, param, period)

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
        xfix=_xfix,
        norm_u=norm_u,
        norm_s=norm_s,
    )

    # Main calculation
    sol = root(func, np.concatenate((xu, xs)))

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
            xfix=_xfix,
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
    with open(sys.argv[1].replace(".json", "_homoclinic.json"), "w") as f:
        res.dump(f)


if __name__ == "__main__":
    _main()
