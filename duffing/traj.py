"""Module for trajectory animation and data dump of system.

This module provides functions to animate the trajectory of the system and dump the trajectory data.
The module works with 2 modes: `animation` and `dump`.

1. Animation mode:
    Animate the trajectory of the system.

2. Dump mode:
    Dump the trajectory data to CSV files.

Example
-------
To animate the trajectory of the system, run the following command:

.. code-block:: bash

        python traj.py data.json

where `data.json` is a JSON file containing the initial state vector, parameter values, and "traj_mode" key set to "animation".

.. code-block:: json

    {
        "x0": [0, 0],
        "parameters": {
            "k": 0.5,
            "B": 0.3,
            "B0": 0.08
        },
        "traj_mode": "animation",
        "traj_animation": {
            "ax_config":{
                "xlim": [-1.5, 1.5],
                "ylim": [-1.5, 1.5]
            },
            "traj_plot_config": {
                "color": "teal"
            },
            "point_plot_config": {
                "color": "orange"
            }
        }
    }

After running the command, the animation window will open. The initial point is set to (0, 0) and the parameter values are set to `k=0.5`, `B0=0.2`, and `B=0.2`.
The following figure shows the animation window:

.. image:: ../_images/ex_traj_animation.png

The animation mode supports the key bindings listed in the documentation for the :func:`traj.traj_animation` function.

.. note::
    "animation" mode accepts the following keys in the ``traj_animation`` dictionary:

    ``ax_config``:
        Configuration of the Axes object.
    ``traj_plot_config``:
        Configuration of the trajectory plot, passed to ``ax.plot()``.
    ``point_plot_config``:
        Configuration of the point plot, passed to ``ax.plot()``.
    ``init_point_plot_config``:
        Configuration of the initial point plot, passed to ``ax.plot()``.
    ``inc_param_keys``:
        Key(s) of the parameter to increment, ``["B", "B0"]`` by default.
    ``inc_param_step``:
        Increment step of the parameter, 0.01 by default.
    ``traj_resolution``:
        Resolution of the trajectory in the period, 100 by default.

To dump the trajectory data to CSV files, run with "dump" mode:

.. code-block:: json

    {
        "x0": [0, 0],
        "parameters": {
            "k": 0.5,
            "B": 0.3,
            "B0": 0.08
        },
        "traj_mode": "dump"
    }


After running the command, the trajectory data will be saved to `data_traj.csv` and `data_poin.csv`.

"""

from typing import Any
import sys, json

import numpy
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

from system import ode_func, Parameter, base_period


def traj_animation(
    fig: Figure,
    ax: Axes,
    vec_x: numpy.ndarray,
    param: Parameter,
    inc_param_keys: str | tuple[str, str],
    inc_param_step: float = 0.01,
    traj_resolution=100,
    traj_plot_config: dict[str, Any] = {},
    point_plot_config: dict[str, Any] = {},
    init_point_plot_config: dict[str, Any] = {},
) -> None:
    """Animate trajectory of ODE system.

    Parameters
    ----------
    fig : Figure
        Figure object.
    ax : Axes
        Axes object.
    vec_x : Any
        Initial state vector.
    param : Parameter
        Parameter object.
    inc_param_keys : str | tuple[str, str]
        Key(s) of the parameter to increment.
    inc_param_step : float, optional
        Increment step of the parameter, by default 0.01.
    traj_resolution : int, optional
        Resolution of the trajectory, by default 50, meaning 50 points per period.
    traj_plot_config : dict[str, Any], optional
        Configuration of the trajectory plot to pass to ax.plot(), by default {"color": "gray", "alpha": 0.5, "linewidth": 0.75}.
    point_plot_config : dict[str, Any], optional
        Configuration of the point plot to pass to ax.plot(), by default {"marker": "o", "color": "r", "markersize": 4}.
    init_point_plot_config : dict[str, Any], optional
        Configuration of the initial point plot to pass to ax.plot(), by default {"color": "g"}.


    Key Bindings
    ------------
    ============ ==================================================
    Key          Action
    ============ ==================================================
    :kbd:`p`     Print current state and parameter values.
    :kbd:`e`     Erase trajectory lines.
    :kbd:`f`     Toggle trajectory plot.
    :kbd:`w`     Increment the first parameter.
    :kbd:`s`     Decrement the first parameter.
    :kbd:`a`     Decrement the second parameter.
    :kbd:`d`     Increment the second parameter.
    :kbd:`=`     Increase increment step.
    :kbd:`-`     Decrease increment step.
    :kbd:`0`     Reset increment step.
    :kbd:`Space` Pause/Resume animation.
    :kbd:`q`     Quit.
    ============ ==================================================

    .. note::
        - Click on the plot to set the initial point.
        - The trajectory is updated by solving the ODE system.
        - The parameter values can be updated by pressing :kbd:`w`, :kbd:`s`, :kbd:`a`, and :kbd:`d`.
        - The increment step can be updated by pressing :kbd:`=`, :kbd:`-`, and :kbd:`0`.

    """

    # Initialization
    _x0 = vec_x.copy()
    t_span = (0, base_period)
    t_eval = np.linspace(*t_span, traj_resolution)
    is_paused = False
    plot_traj = True
    if isinstance(inc_param_keys, str):
        _pk = (inc_param_keys, inc_param_keys)
    else:
        _pk = inc_param_keys
    _param_step = inc_param_step

    traj_plot_config = {
        "color": "gray",
        "alpha": 0.5,
        "linewidth": 0.75,
    } | traj_plot_config
    point_plot_config = {
        "marker": "o",
        "color": "r",
        "markersize": 4,
    } | point_plot_config
    init_point_plot_config = (
        point_plot_config
        | {
            "color": "g",
        }
        | init_point_plot_config
    )

    ax.plot(_x0[0], _x0[1], **init_point_plot_config)

    # Update function for FuncAnimation
    def update(_: int):
        nonlocal _x0
        sol = solve_ivp(ode_func, t_span, _x0, t_eval=t_eval, args=(param,), rtol=1e-8)
        _x0 = sol.y[:, -1]
        if plot_traj:
            new_line = ax.plot(sol.y[0, :], sol.y[1, :], **traj_plot_config)
        else:
            new_line = []
        new_point = ax.plot(_x0[0], _x0[1], **point_plot_config)
        return new_line + new_point

    # Event handlers
    ## Key press
    def on_press(event):
        print("press ", event.key, "| ", end="")
        nonlocal is_paused, _x0, init_point_plot_config, _param_step, plot_traj

        def erase_lines(say="Erased."):
            for l in list(ax.lines):
                l.remove()
            ax.plot(_x0[0], _x0[1], **init_point_plot_config)
            print(say)

        match event.key:
            case "p":
                print(
                    "x0:",
                    np.array2string(_x0, separator=", ", sign="+"),
                    f"Parameter: {param}",
                )
            case "e":
                erase_lines()
            case "f":
                plot_traj = not plot_traj
                erase_lines(f"Plot trajectory: {plot_traj}")
            case "w" | "a" | "s" | "d":
                _i = 0 if event.key in ("a", "d") else 1
                _s = _param_step if event.key in ("d", "w") else -_param_step
                param.increment(_pk[_i], _s)
                erase_lines(f"Parameter updated: {param}")
            case "=" | "-" | "0":
                if event.key == "=":
                    _param_step *= 10
                elif event.key == "-":
                    _param_step /= 10
                else:
                    _param_step = inc_param_step
                print(f"Increment step updated: {_param_step:.2e}")
            case " ":
                is_paused = not is_paused
                if is_paused:
                    ani.event_source.stop()
                else:
                    ani.event_source.start()
                print("Paused." if is_paused else "Resumed.")
            case "q":
                print("Quit.")
                sys.exit()
            case _:
                print("No action for the key.")

        sys.stdout.flush()

    ## Mouse click
    def on_click(event):
        print(f"click: {event.xdata}, {event.ydata}")
        nonlocal _x0
        _x0 = np.array([event.xdata, event.ydata])
        ax.plot(_x0[0], _x0[1], **init_point_plot_config)
        sys.stdout.flush()

    # Connect event handlers
    if (manager := fig.canvas.manager) is not None and (
        hid := manager.key_press_handler_id
    ) is not None:
        fig.canvas.mpl_disconnect(hid)  # Disconnect default key press handler
    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Animation
    ani = FuncAnimation(fig, update, frames=100, interval=50)
    plt.show()


def dump_trajectory(
    vec_x: numpy.ndarray,
    param: Parameter,
    traj_resolution: int = 100,
    itr_max: int = 100,
) -> dict[str, numpy.ndarray]:
    """Dump trajectory data of ODE system.

    Parameters
    ----------
    vec_x : numpy.ndarray
        Initial state vector.
    param : Parameter
        Parameter object.
    traj_resolution : int, optional
        Resolution of the trajectory, by default 100, meaning 100 points per period.
    itr_max : int, optional
        Maximum number of iterations, by default 1000.

    Returns
    -------
    dict[str, numpy.ndarray]
        Trajectory data. Keys are "traj" and "points".
    """
    t_span = (0, base_period)
    t_eval = np.linspace(*t_span, traj_resolution)
    traj = np.zeros((itr_max * traj_resolution, 3))
    points = np.zeros((itr_max, 2))

    _x0 = vec_x.copy()
    for i in range(itr_max):
        sol = solve_ivp(ode_func, t_span, _x0, t_eval=t_eval, args=(param,), rtol=1e-8)
        traj[i * traj_resolution : (i + 1) * traj_resolution, 0] = (
            sol.t + i * base_period
        )
        traj[i * traj_resolution : (i + 1) * traj_resolution, 1:] = sol.y.T
        points[i] = _x0
        _x0 = sol.y[:, -1]

    return {
        "traj": traj.tolist(),
        "points": points.tolist(),
    }


def _main():
    # Load data from JSON file
    try:
        with open(sys.argv[1], "r") as f:
            data = json.load(f)
        x0 = np.array(data.get("x0", [0, 0]))
        param = Parameter(**data.get("parameters", {}))
        mode = data.get("traj_mode", "animation")

    except IndexError:
        raise IndexError("Usage: python traj.py [data.json]")
    except FileNotFoundError:
        raise FileNotFoundError(f"{sys.argv[1]} not found")

    if mode == "animation":
        config = {
            "vec_x": x0,
            "param": param,
            "inc_param_keys": ["B", "B0"],
            "inc_param_step": 0.01,
            "traj_resolution": 100,
            "traj_plot_config": {},
            "point_plot_config": {},
            "init_point_plot_config": {},
        }
        for key in config:
            if key in data.get("traj_animation", {}):
                config[key] = data["traj_animation"][key]

        ax_config = {
            "xlim": (-1, 1),
            "ylim": (-1, 1),
            "aspect": "equal",
            "xlabel": "x",
            "ylabel": "y",
        }
        if "ax_config" in data.get("traj_animation", {}):
            ax_config.update(data["traj_animation"]["ax_config"])

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set(**ax_config)
        ax.grid()

        traj_animation(fig=fig, ax=ax, **config)

    elif mode == "dump":
        traj_file = sys.argv[1].replace(".json", "_traj.csv")
        poin_file = sys.argv[1].replace(".json", "_poin.csv")
        res = dump_trajectory(x0, param)

        with open(traj_file, "w") as f:
            np.savetxt(f, res["traj"], delimiter=",")
        with open(poin_file, "w") as f:
            np.savetxt(f, res["points"], delimiter=",")
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    _main()
