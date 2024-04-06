"""Module to calculate homoclinic bifurcation point.

Examples
--------
Prepare a JSON file with the following content:

.. code-block:: json

    {
        "xfix": [-1.003308564506079, 0.29656370772365503],
        "xu": [-1.0033042799251857, 0.29643129025277903],
        "xs": [-1.0031997049668901, 0.2967252732614351],
        "parameters": {
            "k": 0.05,
            "B0": 0,
            "B": 0.225
        },
        "period": 1,
        "maps_u": 7,
        "maps_s": 6,
        "hbf_param_key": "B"
    }

Run the following command in your terminal with the JSON file:

.. code-block:: bash

    python hbf.py [data.json]

The result will be printed and dumped to a JSON file with the same name as the input file but with "_hbf" suffix.
The content of the dumped file will be like below:

.. code-block:: json

    {
        "xfix": [
            -1.0045625374115001,
            0.3021901997786167
        ],
        "xu": [
            -1.0045590724479194,
            0.30205195722416683
        ],
        "xs": [
            -1.0044433101609014,
            0.30236677831803677
        ],
        "parameters": {
            "k": 0.05,
            "B": 0.22194620514454416,
            "B0": 0
        },
        "period": 1,
        "maps_u": 7,
        "maps_s": 6,
        "hbf_param_key": "B"
    }

Using :py:mod:`manifold` module, you can draw the stable and unstable manifolds of the homoclinic tangency.

.. image:: ../_images/ex_hbf_manifold.png

See Also
--------
homoclinic : Homoclinic point calculation.
manifold: Manifold module, which contains functions to draw the stable and unstable manifolds.

"""

import sys, json
from collections.abc import Callable
import numpy
import numpy as np
from scipy.optimize import root

from core import IterItems
from manifold import prepare_by_fix
from system import Parameter, PoincareMapResult, poincare_map
from fix import fix, fix_func
from homoclinic import homoclinic, homoclinic_func, calc_tvec_diff


class HbfResult(IterItems):
    """Homoclinic bifurcation point calculation result.

    Attributes
    ----------
    success : bool
        Success flag.
    message : str
        Message of the calculation result.
    xu : numpy.ndarray
        Homoclinic point in the unstable manifold.
    xs : numpy.ndarray
        Homoclinic point in the stable manifold.
    xh : numpy.ndarray
        Homoclinic point.
    xh_err : numpy.ndarray
        Error of the homoclinic point, calculated as xh_u - xh_s.
    xfix : numpy.ndarray
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
        xfix: numpy.ndarray | None = None,
        period: int | None = None,
        xu: numpy.ndarray | None = None,
        xs: numpy.ndarray | None = None,
        xh: numpy.ndarray | None = None,
        xh_err: numpy.ndarray | None = None,
        tvec_diff: float | None = None,
        parameters: Parameter | None = None,
        maps_u: int | None = None,
        maps_s: int | None = None,
        hbf_param_key: str | None = None,
    ) -> None:
        self.success = success
        self.message = message
        self.xfix = xfix
        self.period = period
        self.xu = xu
        self.xs = xs
        self.xh = xh
        self.xh_err = xh_err
        self.tvec_diff = tvec_diff
        self.parameters = parameters
        self.maps_u = maps_u
        self.maps_s = maps_s
        self.hbf_param_key = hbf_param_key
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
        return f"HbfResult(\n{'\n'.join(self.out_strs())}\n)"


def hbf_func(
    vars: numpy.ndarray,
    period: int,
    pmap: Callable[[numpy.ndarray, Parameter], PoincareMapResult],
    pmap_u: Callable[[numpy.ndarray, Parameter], PoincareMapResult],
    pmap_s: Callable[[numpy.ndarray, Parameter], PoincareMapResult],
    param: Parameter,
    hbf_param_key: str,
    verbose: bool = False,
) -> numpy.ndarray:
    """Function for evaluating homoclinic bifurcation point.

    Parameters
    ----------
    vars : numpy.ndarray
        Initial points of the fixed or periodic point, homoclinic point in the unstable manifold, homoclinic point in the stable manifold, and bifurcation parameter. The shape of the array is (7,).
    period : int
        Period of the periodic point. For fixed point, set 1.
    pmap : Callable[[numpy.ndarray, Parameter], PoincareMapResult]
        Poincare map function.
    pmap_u : Callable[[numpy.ndarray, Parameter], PoincareMapResult]
        Function to map from [xu, yu] to [xh, yh].
    pmap_s : Callable[[numpy.ndarray, Parameter], PoincareMapResult]
        Function to map from [xs, ys] to [xh, yh].
    param : Parameter
        Parameter object.
    hbf_param_key : str
        Bifurcation parameter key to control.
    verbose : bool, optional
        Print progress, by default False.

    Returns
    -------
    numpy.ndarray
        Residual vector.

    Raises
    ------
    ValueError
        If the fixed point calculation fails.
    """
    xfix0 = vars[0:2]
    x_u0 = vars[2:4]
    x_s0 = vars[4:6]
    setattr(param, hbf_param_key, vars[6])

    xfix, u_evec, s_evec = prepare_by_fix(xfix0, param, period)[0:3]

    norm_mat = np.array([[0, 1], [-1, 0]])
    nvec_u = norm_mat @ u_evec
    nvec_s = norm_mat @ s_evec

    _pmap = lambda x: pmap(x, param).x
    _pmap_u = lambda x: pmap_u(x, param).x
    _pmap_s = lambda x: pmap_s(x, param).x

    if (jac_u := pmap_u(x_u0, param).jac) is None:
        raise ValueError("Failed to calculate Jacobian matrix of pmap_u.")
    if (jac_s := pmap_s(x_s0, param).jac) is None:
        raise ValueError("Failed to calculate Jacobian matrix of pmap_s.")

    ret = np.empty(7)
    ret[0:2] = fix_func(xfix0, _pmap)
    ret[2:6] = homoclinic_func(vars[2:6], _pmap_u, _pmap_s, xfix, nvec_u, nvec_s)
    ret[6] = calc_tvec_diff(jac_u, jac_s, u_evec, s_evec)

    if verbose:
        print("Running... :", [f"{r:+.3e}" for r in ret], end="\r")
    return ret


def hbf(
    xfix0: numpy.ndarray,
    period: int,
    param: Parameter,
    hbf_param_key: str,
    xu0: numpy.ndarray,
    xs0: numpy.ndarray,
    maps_u: int,
    maps_s: int,
    verbose: bool = False,
) -> HbfResult:
    """Homoclinic point calculation.

    Parameters
    ----------
    xfix0 : numpy.ndarray
        Fixed or periodic point.
    period : int
        Period of the fixed or periodic point.
    param : Parameter
        Parameter object.
    hbf_param_key : str
        Bifurcation parameter key to control.
    xu0 : numpy.ndarray
        Initial point of the homoclinic point in the unstable manifold.
    xs0 : numpy.ndarray
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

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from hbf import Parameter, hbf

        hbf_args = {
            "xfix0": np.array([-1.003308564506079, 0.29656370772365503]),
            "period": 1,
            "param": Parameter(k=0.05, B0=0, B=0.225),
            "hbf_param_key": "B",
            "xu0": np.array([-1.0033042799251857, 0.29643129025277903]),
            "xs0": np.array([-1.0031997049668901, 0.2967252732614351]),
            "maps_u": 7,
            "maps_s": 6,
            "verbose": True,
        }

        res = hbf(**hbf_args)
        print(res)

    The above code will print the homoclinic bifurcation point information like below:

    .. code-block:: python

        HbfResult(
            success: True
            message: Success
            xfix: [-1.00456254  0.3021902 ]
            period: 1
            xu: [-1.00455907  0.30205196]
            xs: [-1.00444331  0.30236678]
            xh: [-0.16507173 -0.23115691]
            xh_err: [-1.92765989e-09  9.95684063e-10]
            tvec_diff: -2.422641183612425e-05
            parameters: (k, B0, B) = (+0.050000, +0.000000, +0.221946)
            maps_u: 7
            maps_s: 6
            hbf_param_key: B
        )

    See Also
    --------
    homoclinic : Homoclinic point calculation.
    manifold: Manifold module, which contains functions to draw the stable and unstable manifolds.
    """

    # Define the maps
    _, _, _, u_itr_cnt, s_itr_cnt = prepare_by_fix(xfix0, param, period)

    pmap = lambda x, p: poincare_map(x, p, calc_jac=True)
    pmap_u = lambda x, p: poincare_map(x, p, itr_cnt=u_itr_cnt * maps_u, calc_jac=True)
    pmap_s = lambda x, p: poincare_map(
        x, p, itr_cnt=s_itr_cnt * maps_s, calc_jac=True, inverse=True
    )

    # Define the function for root finding
    func = lambda x: hbf_func(
        x, period, pmap, pmap_u, pmap_s, param, hbf_param_key, verbose
    )

    # Initial guess
    vars = np.empty(7)
    vars[0:6] = np.concatenate((xfix0, xu0, xs0))
    vars[6] = getattr(param, hbf_param_key)

    # Main calculation
    sol = root(func, vars)
    if verbose:
        print()

    # Post-process
    if sol.success:
        xfix = sol.x[0:2]
        xu = sol.x[2:4]
        xs = sol.x[4:6]
        setattr(param, hbf_param_key, sol.x[6])

        fix_result = fix(xfix, param, period)
        if (xfix := fix_result.xfix) is None:
            raise ValueError("Fixed point calculation failed.")

        homo_result = homoclinic(xfix, period, param, xu, xs, maps_u, maps_s)
        if not homo_result.success:
            raise ValueError("Homoclinic point calculation failed.")
        if (xu := homo_result.xu) is None:
            raise ValueError("Failed to find xu.")
        if (xs := homo_result.xs) is None:
            raise ValueError("Failed to find xs.")
        if (xh := homo_result.xh) is None:
            raise ValueError("Failed to find xh.")
        if (xh_err := homo_result.xh_err) is None:
            raise ValueError("Failed to calculate xh_err.")
        if (tvec_diff := homo_result.tvec_diff) is None:
            raise ValueError("Failed to calculate tvec_diff.")

        return HbfResult(
            success=True,
            message="Success",
            xu=xu,
            xs=xs,
            xh=xh,
            xh_err=xh_err,
            xfix=xfix,
            tvec_diff=tvec_diff,
            parameters=param,
            maps_u=maps_u,
            maps_s=maps_s,
            hbf_param_key=hbf_param_key,
            period=period,
        )
    else:
        return HbfResult(success=False, message=sol.message)


def __main():
    # Load data from JSON file
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
        hbf_param_key = data.get("hbf_param_key", "B")
    except IndexError:
        raise IndexError("Usage: python fix.py [data.json]")
    except FileNotFoundError:
        raise FileNotFoundError(f"{sys.argv[1]} not found")

    # Calculate homoclinic bifurcation point
    res = hbf(x0, period, param, hbf_param_key, xu0, xs0, maps_u, maps_s, verbose=True)
    print(res)

    # Dump the result to a JSON file
    with open(sys.argv[1].replace(".json", "_hbf.json"), "w") as f:
        res.dump(f)


if __name__ == "__main__":
    __main()
