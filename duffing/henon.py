from typing import Any

import numpy
import numpy as np
from core import IterItems


class Parameter(IterItems):
    def __init__(self, a: float = 0.2, b: float = 0.2) -> None:
        self.a = a
        self.b = b
        super().__init__(["a", "b"])

    def __repr__(self) -> str:
        return f"(a, b) = ({self.a:+.6f}, {self.b:+.6f})"

    def increment(self, key: str, step: float = 0.01) -> None:
        setattr(self, key, getattr(self, key) + step)


def henon_map(
    vec_x: numpy.ndarray,
    param: Parameter,
    inverse: bool = False,
) -> numpy.ndarray:
    x, y = vec_x[0:2]
    a, b = param.a, param.b
    if not inverse:
        return numpy.array([1 - a * x**2 + y, b * x])
    else:
        return numpy.array([y / b, x + (a * y**2 / b**2) - 1])


def henon_jac(
    vec_x: numpy.ndarray,
    param: Parameter,
    inverse: bool = False,
) -> numpy.ndarray:
    x, y = vec_x[0:2]
    a, b = param.a, param.b

    if not inverse:
        return numpy.array([[-2 * a * x, 1], [b, 0]])
    else:
        return numpy.array([[0, 1 / b], [1, 2 * a * y / b**2]])


def henon_jac_p(
    vec_x: numpy.ndarray, param: Parameter, inverse: bool = False
) -> dict[str, numpy.ndarray]:
    x, y = vec_x[0:2]
    a, b = param.a, param.b

    if not inverse:
        return {
            "a": np.array([-(x**2), 0]),
            "b": np.array([0, x]),
        }
    else:
        return {
            "a": np.array([0, y**2 / b**2]),
            "b": np.array([1 / b, -2 * a * y**2 / b**3]),
        }


class PoincareMapResult:
    """Result of the Poincare map calculation.

    Attributes
    ----------
    x : numpy.ndarray
        Image of x0 under the Poincare map. See :func:`system.poincare_map`.
    jac : numpy.ndarray | None
        Jacobian matrix of the Poincare map, if calculated.
    pjac : numpy.ndarray | None
        Jacobian matrix of the Poincare map with respect to the parameter, if calculated.
    pjac_key : str | None
        Key of the parameter to calculate Jacobian matrix.

    See Also
    --------
    poincare_map
    """

    def __init__(
        self,
        x: numpy.ndarray,
        jac: numpy.ndarray | None = None,
        pjac: numpy.ndarray | None = None,
        str_pjac_key: str | None = None,
    ) -> None:
        self.x = x
        self.jac = jac
        self.pjac = pjac
        self.pjac_key = str_pjac_key

    def __repr__(self) -> str:
        return f"PoincareMapResult({self.x=}, {self.jac=}, {self.pjac=})"


def poincare_map(
    x0: Any,
    param: Parameter,
    itr_cnt: int = 1,
    calc_jac: bool = False,
    pjac_key: str | None = None,
    inverse: bool = False,
) -> PoincareMapResult:
    """Calculate the Poincare map.

    Parameters
    ----------
    x0 : Any
        Initial state of the system.
    param : Parameter
        Parameters of the system.
    itr_cnt : int, optional
        Number of iterations, by default 1
    calc_jac : bool, optional
        Whether to calculate the Jacobian matrix, by default False
    pjac_key : str, optional
        Key of the parameter to calculate the Jacobian matrix, by default None
    inverse : bool, optional
        Whether to calculate the inverse map, by default False

    Returns
    -------
    PoincareMapResult
        Result of the Poincare map calculation.

    See Also
    --------
    PoincareMapResult
    """
    if not calc_jac:
        func = lambda _x: henon_map(_x, param, inverse)
        _x0 = x0.copy()
    else:

        def func(_x):
            ret = np.hstack(
                [
                    henon_map(_x, param, inverse),
                    henon_jac(_x, param, inverse).flatten(order="F"),
                ]
            ).flatten()
            if pjac_key is not None:
                ret = np.hstack(
                    [ret, henon_jac_p(_x, param, inverse)[pjac_key].flatten(order="F")]
                )
            return ret

        _x0 = x0.copy()
        jac = np.eye(2)

        if pjac_key is not None:
            jac_p = np.zeros(2)

    for _ in range(itr_cnt):
        sol = func(_x0)

        _x0 = sol[:2]
        if calc_jac:
            jack = sol[2:6].reshape(2, 2, order="F")

            if pjac_key is not None:
                jac_p = jack @ jac_p + sol[6:8]

            jac = jack @ jac

    ret = PoincareMapResult(_x0[:2])

    if calc_jac:
        ret.jac = jac

        if pjac_key is not None:
            ret.pjac = jac_p
            ret.pjac_key = pjac_key

    return ret
