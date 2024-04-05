"""Module to calculate fixed point.

Examples
--------
Prepare a JSON file with the following content:

.. code-block:: json

    {
        "xfix": [0.0, 0.0],
        "parameters": {
            "k": 0.2,
            "B": 0.1,
            "B0": 0.1
        },
        "period": 1
    }

Run the following command in your terminal with the JSON file:

.. code-block:: bash

    python fix.py [data.json]

The result will be printed and dumped to a JSON file with the same name as the input file but with "_fix" suffix.
The content of the dumped file will be like below:

.. code-block:: json

    {
        "xfix": [
            0.23214784788536827,
            0.07559285853639049
        ],
        "parameters": {
            "k": 0.2,
            "B": 0.1,
            "B0": 0.1
        },
        "period": 1
    }

"""

import sys, json
from collections.abc import Callable
import numpy
import numpy as np
from scipy.optimize import root

from core import IterItems
from system import poincare_map, Parameter


class FixResult(IterItems):
    """Fixed or periodic point calculation result.

    Attributes
    ----------
    success : bool
        Success flag.
    message : str
        Message of the calculation result.
    xfix : numpy.ndarray | None
        Fixed or periodic point.
    eig : numpy.ndarray | None
        Eigenvalues of the Jacobian matrix of the Poincare map at the fixed or periodic point.
    abs_eig : numpy.ndarray | None
        Absolute values of the eigenvalues.
    evec : numpy.ndarray | None
        Eigenvectors of the Jacobian matrix.
    u_edim : int | None
        Dimension of the unstable eigenspace.
    c_edim : int | None
        Dimension of the center eigenspace.
    s_edim : int | None
        Dimension of the stable eigenspace.
    u_eig : numpy.ndarray | None
        Unstable eigenvalues.
    c_eig : numpy.ndarray | None
        Center eigenvalues.
    s_eig : numpy.ndarray | None
        Stable eigenvalues.
    u_evec : numpy.ndarray | None
        Unstable eigenvectors.
    c_evec : numpy.ndarray | None
        Center eigenvectors.
    s_evec : numpy.ndarray | None
        Stable eigenvectors.
    """

    def __init__(
        self,
        success: bool,
        message: str,
        xfix: numpy.ndarray | None = None,
        parameters: Parameter | None = None,
        period: int | None = None,
        eig: numpy.ndarray | None = None,
        abs_eig: numpy.ndarray | None = None,
        evec: numpy.ndarray | None = None,
        u_edim: int | None = None,
        c_edim: int | None = None,
        s_edim: int | None = None,
        u_eig: numpy.ndarray | None = None,
        c_eig: numpy.ndarray | None = None,
        s_eig: numpy.ndarray | None = None,
        u_evec: numpy.ndarray | None = None,
        c_evec: numpy.ndarray | None = None,
        s_evec: numpy.ndarray | None = None,
    ):
        self.success = success
        self.message = message
        self.xfix = xfix
        self.parameters = parameters
        self.period = period
        self.eig = eig
        self.abs_eig = abs_eig
        self.evec = evec
        self.u_edim = u_edim
        self.c_edim = c_edim
        self.s_edim = s_edim
        self.u_eig = u_eig
        self.c_eig = c_eig
        self.s_eig = s_eig
        self.u_evec = u_evec
        self.c_evec = c_evec
        self.s_evec = s_evec
        super().__init__(["xfix", "parameters", "period"])

    def __repr__(self) -> str:
        return f"FixResult(\n{'\n'.join(self.out_strs())}\n)"


def fix_func(
    vec_x: numpy.ndarray, pmap: Callable[[numpy.ndarray], numpy.ndarray]
) -> numpy.ndarray:
    """Function for evaluating fixed or periodic point.

    Parameters
    ----------
    vec_x : numpy.ndarray
        Initial point.
    pmap : Callable[[numpy.ndarray], numpy.ndarray]
        Poincare map function.

    Returns
    -------
    numpy.ndarray
        Residual vector.
    """
    return pmap(vec_x) - vec_x


def fix(vec_x: numpy.ndarray, param: Parameter, period: int = 1) -> FixResult:
    """Fixed or periodic point calculation using the root method of SciPy (quasi-Newton's method).

    Parameters
    ----------
    vec_x : numpy.ndarray
        Initial state vector of the fixed or periodic point.
    param : Parameter
        Parameter object.
    period : int
        Period of the target periodic point, by default 1, meaning the fixed point.

    Returns
    -------
    FixResult
        Fixed or periodic point information.

    Examples
    --------

    .. code-block:: python

        from fix import fix
        from system import Parameter

        x0 = [0, 0]
        param = Parameter(k=0.2, B=0.1, B0=0.1)
        period = 1
        res = fix(x0, param, period)

        print(res)

    The above code will print the fixed point information like below:

    .. code-block:: python

        FixResult(
            success: True
            message: Success
            xfix: [0.23214785 0.07559286]
            parameters: (k, B0, B) = (+0.200000, +0.100000, +0.100000)
            period: 1
            eig: [-0.21783035+0.48698937j -0.21783035-0.48698937j]
            abs_eig: [0.53348731 0.53348731]
            evec: [[ 0.9416779 +0.j          0.9416779 -0.j        ]
        [-0.10493791-0.31973545j -0.10493791+0.31973545j]]
            u_edim: 0
            c_edim: 0
            s_edim: 2
            u_eig: []
            c_eig: []
            s_eig: [-0.21783035+0.48698937j -0.21783035-0.48698937j]
            u_evec: []
            c_evec: []
            s_evec: [[ 0.9416779 +0.j          0.9416779 -0.j        ]
        [-0.10493791-0.31973545j -0.10493791+0.31973545j]]
        )
    """
    x0 = vec_x.copy()
    pmap = lambda x: poincare_map(x, param, itr_cnt=period).x
    func = lambda x: fix_func(x, pmap)

    sol = root(func, x0)

    if sol.success:
        xfix = sol.x
        if (
            jac := poincare_map(xfix, param, itr_cnt=period, calc_jac=True).jac
        ) is None:
            raise ValueError(
                "Jacobian matrix is not calculated though fix is successful."
            )
        eig, vec = np.linalg.eig(jac)

        u_edim = np.sum(np.abs(np.real(eig)) > 1, dtype=int)
        c_edim = np.sum(np.abs(np.real(eig)) == 1, dtype=int)
        s_edim = np.sum(np.abs(np.real(eig)) < 1, dtype=int)

        u_eigs = eig[np.abs(np.real(eig)) > 1]
        c_eigs = eig[np.abs(np.real(eig)) == 1]
        s_eigs = eig[np.abs(np.real(eig)) < 1]

        u_evecs = vec[:, np.abs(np.real(eig)) > 1]
        c_evecs = vec[:, np.abs(np.real(eig)) == 1]
        s_evecs = vec[:, np.abs(np.real(eig)) < 1]

        return FixResult(
            success=True,
            xfix=xfix,
            eig=eig,
            abs_eig=np.abs(eig),
            evec=vec,
            u_edim=u_edim,
            c_edim=c_edim,
            s_edim=s_edim,
            u_eig=u_eigs,
            c_eig=c_eigs,
            s_eig=s_eigs,
            u_evec=u_evecs,
            c_evec=c_evecs,
            s_evec=s_evecs,
            period=period,
            parameters=param,
            message="Success",
        )
    else:
        return FixResult(success=False, message=sol.message)


def _main():
    # Load data from JSON file
    try:
        with open(sys.argv[1], "r") as f:
            data = json.load(f)
        x0 = np.array(data.get("xfix", [0, 0]))
        param = Parameter(**data.get("parameters", {}))
        period = data.get("period", 1)
    except IndexError:
        raise IndexError("Usage: python fix.py [data.json]")
    except FileNotFoundError:
        raise FileNotFoundError(f"{sys.argv[1]} not found")

    # Calculate fixed or periodic point
    res = fix(x0, param, period)

    # Print the result
    print(res)

    # Dump the result to a file
    with open(sys.argv[1].replace(".json", "_fix.json"), "w") as f:
        res.dump(f)


if __name__ == "__main__":
    _main()
