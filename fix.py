"""Module to calculate fixed point."""

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
    xfix : numpy.ndarray
        Fixed or periodic point.
    eig : numpy.ndarray
        Eigenvalues of the Jacobian matrix of the Poincare map at the fixed or periodic point.
    abs_eig : numpy.ndarray
        Absolute values of the eigenvalues.
    evec : numpy.ndarray
        Eigenvectors of the Jacobian matrix.
    u_edim : int
        Number of unstable eigenvalues.
    c_edim : int
        Number of center eigenvalues.
    s_edim : int
        Number of stable eigenvalues.
    u_eig : numpy.ndarray
        Unstable eigenvalues.
    c_eig : numpy.ndarray
        Center eigenvalues.
    s_eig : numpy.ndarray
        Stable eigenvalues.
    u_evec : numpy.ndarray
        Unstable eigenvectors.
    c_evec : numpy.ndarray
        Center eigenvectors.
    s_evec : numpy.ndarray
        Stable eigenvectors.
    message : str
        Message of the calculation result.
    """

    def __init__(
        self,
        success: bool,
        message: str,
        xfix: numpy.ndarray | None = None,
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
        period: int | None = None,
        parameters: Parameter | None = None,
    ):
        self.success = success
        self.message = message
        self.xfix = xfix
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
        self.period = period
        self.parameters = parameters
        super().__init__(["xfix", "parameters", "period"])

    def __repr__(self) -> str:
        return f"FixResult({self.success=}, {self.xfix=}, {self.eig=}, {self.abs_eig=}, {self.evec=}, {self.u_edim=}, {self.c_edim=}, {self.s_edim=}, {self.u_eig=}, {self.c_eig=}, {self.s_eig=}, {self.u_evec=}, {self.c_evec=}, {self.s_evec=}, {self.message=})"


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
    """Fixed or periodic point calculation.

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
    print(dict(res))
    res.dump(sys.stdout)


if __name__ == "__main__":
    _main()
