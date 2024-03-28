from typing import Any
import numpy as np
from scipy.integrate import solve_ivp


class Parameter:
    """Parameter class for Duffing oscillator."""

    def __init__(self, k: float = 0.2, B: float = 0.1, B0: float = 0.1) -> None:
        self.k = k
        self.B = B
        self.B0 = B0

    def __repr__(self) -> str:
        return f"(k, B0, B) = ({self.k:+.6f}, {self.B0:+.6f}, {self.B:+.6f})"

    def increment(self, key: str, step: float = 0.01) -> None:
        """Increment the parameter value.

        Parameters
        ----------
        key : str
            Key of the parameter to increment.
        step : float, optional
            Increment step, by default 0.01.
        """
        setattr(self, key, getattr(self, key) + step)


def ode_func(t: float, vec_x: np.ndarray, param: Parameter) -> np.ndarray:
    """Duffing oscillator ODE function.

    Parameters
    ----------
    t : float
        Time.
    vec_x : np.ndarray
        State vector.
    param : Parameter
        Parameter object.

    Returns
    -------
    np.ndarray
        Derivative of the state vector.
    """
    x, y = vec_x
    k, B, B0 = param.k, param.B, param.B0

    dxdt = y
    dydt = -k * y - x**3 + B0 + B * np.cos(t)

    return np.array([dxdt, dydt])


def ode_func_jac_x(t: float, vec_x: np.ndarray, param: Parameter) -> np.ndarray:
    """Jacobian of the Duffing oscillator ODE function with respect to state vector.

    Parameters
    ----------
    t : float
        Time.
    vec_x : np.ndarray
        State vector.
    param : Parameter
        Parameter object.

    Returns
    -------
    np.ndarray
        Jacobian matrix.
    """
    x = vec_x[0]
    k = param.k

    return np.array([[0, 1], [-3 * x**2, -k]])


def ode_func_jac_p(
    t: float, vec_x: np.ndarray, param: Parameter
) -> dict[str, np.ndarray]:
    """Jacobian of the Duffing oscillator ODE function with respect to parameter.

    Parameters
    ----------
    t : float
        Time.
    vec_x : np.ndarray
        State vector.
    param : Parameter
        Parameter object.

    Returns
    -------
    dict[str, np.ndarray]
        Jacobian matrix for each parameter.
    """

    y = vec_x[1]

    return {
        "k": np.array([0, -y]),
        "B0": np.array([0, 1]),
        "B": np.array([0, np.cos(t)]),
    }


def ode_func_calc_jac(
    t: float, vec_x: np.ndarray, param: Parameter, param_key: str | None = None
) -> np.ndarray:
    """ODE function for calculating Jacobian matrix with respect to initial state and parameter.

    Parameters
    ----------
    t : float
        Time.
    vec_x : np.ndarray
        State vector.
    param : Parameter
        Parameter object.
    param_key : str | None, optional
        Key of the parameter to calculate Jacobian matrix, by default None.

    Returns
    -------
    np.ndarray
        Derivative of the Jacobian matrix.
    """

    phi = vec_x[0:2]
    dPhidX0 = vec_x[2:].reshape(2, 3, order="F")

    ret = ode_func_jac_x(t, phi, param) @ dPhidX0
    if param_key is not None:
        ret[:, 2] += ode_func_jac_p(t, phi, param)[param_key]

    return ret.flatten(order="F")


base_period = 2 * np.pi


def poincare_map(
    vec_x: Any,
    param: Parameter,
    itr_cnt: int = 1,
    calc_jac: bool = False,
    pjac_key: str | None = None,
    inverse: bool = False,
) -> dict[str, np.ndarray]:
    """Poincare map for Duffing oscillator.

    Parameters
    ----------
    vec_x : Any
        State vector.
    param : Parameter
        Parameter object.
    itr_cnt : int, optional
        Iteration count of the maps, by default 1.
    calc_jac : bool, optional
        Whether to calculate Jacobian matrix or not, by default False.
    pjac_key : str | None, optional
        Key of the parameter to calculate Jacobian matrix, by default None.

    Returns
    -------
    dict[str, np.ndarray]
        Result of the Poincare map. If `calc_jac` is True, Jacobian matrix is also returned.
        If `pjac_key` is not None, Jacobian matrix with respect to the parameter is also returned.
    """
    # Setup variables
    if not calc_jac:
        func = lambda _t, _x: ode_func(_t, _x, param)
        x0 = vec_x.copy()
    else:
        func = lambda _t, _x: np.hstack(
            [ode_func(_t, _x[0:2], param), ode_func_calc_jac(_t, _x, param, pjac_key)]
        ).flatten()
        x0 = np.hstack([vec_x, np.eye(2).flatten(), np.zeros(2)])
        jac = np.eye(2)

        if pjac_key is not None:
            jac_p = np.zeros(2)

    # Calculate the Poincare map
    for _ in range(itr_cnt):
        sol = solve_ivp(
            func, [0, base_period if not inverse else -base_period], x0, rtol=1e-8
        )

        x0[:2] = sol.y[:2, -1]
        if calc_jac:
            jack = sol.y[2:6, -1].reshape(2, 2, order="F")

            if pjac_key is not None:
                jac_p = jack @ jac_p + sol.y[6:8, -1]

            jac = jack @ jac

    # Return the result
    ret = {"x": np.array(x0[:2])}
    if calc_jac:
        ret["jac"] = jac

        if pjac_key is not None:
            ret[f"jac_p({pjac_key})"] = jac_p

    return ret


poincare_map(
    [0.23252517, 0.07522187], Parameter(), itr_cnt=2, calc_jac=True, pjac_key="B"
)
