from typing import Any
import sys, json

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

from duffing import ode_func, Parameter, base_period


def traj_animation(
    fig: Figure,
    ax: Axes,
    vec_x: np.ndarray,
    param: Parameter,
    inc_param_keys: str | tuple[str, str],
    inc_param_step: float = 0.01,
    itr_max=1000,
    traj_resolution=100,
    traj_plot_config: dict[str, Any] = {
        "color": "gray",
        "alpha": 0.5,
        "linewidth": 0.75,
    },
    point_plot_config: dict[str, Any] = {
        "marker": "o",
        "color": "r",
        "markersize": 4,
    },
    init_point_plot_config: dict[str, Any] = {
        "color": "g",
    },
) -> None:
    """Animate trajectory of Duffing oscillator.

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
    itr_max : int, optional
        Maximum number of iterations, by default 1000.
    traj_resolution : int, optional
        Resolution of the trajectory, by default 50, meaning 50 points per period.
    traj_plot_config : dict[str, Any], optional
        Configuration of the trajectory plot to pass to ax.plot(), by default {"color": "gray", "alpha": 0.5, "linewidth": 0.75}.
    point_plot_config : dict[str, Any], optional
        Configuration of the point plot to pass to ax.plot(), by default {"marker": "o", "color": "r", "markersize": 4}.
    init_point_plot_config : dict[str, Any], optional
        Configuration of the initial point plot to pass to ax.plot(), by default {"color": "g"}.
    """

    # Initialization
    _x0 = vec_x.copy()
    t_span = (0, base_period)
    t_eval = np.linspace(*t_span, traj_resolution)
    is_paused = False
    if isinstance(inc_param_keys, str):
        _pk = (inc_param_keys, inc_param_keys)
    else:
        _pk = inc_param_keys

    init_point_plot_config = point_plot_config | init_point_plot_config
    ax.plot(_x0[0], _x0[1], **init_point_plot_config)

    # Update function for FuncAnimation
    def update(_: int):
        nonlocal _x0
        sol = solve_ivp(ode_func, t_span, _x0, t_eval=t_eval, args=(param,), rtol=1e-8)
        _x0 = sol.y[:, -1]
        new_line = ax.plot(sol.y[0, :], sol.y[1, :], **traj_plot_config)
        new_point = ax.plot(_x0[0], _x0[1], **point_plot_config)
        return new_line + new_point

    # Event handlers
    ## Key press
    def on_press(event):
        print("press ", event.key, "| ", end="")
        nonlocal is_paused, _x0, init_point_plot_config

        def erase_lines():
            for l in list(ax.lines):
                l.remove()
            ax.plot(_x0[0], _x0[1], **init_point_plot_config)

        match event.key:
            case "p":
                print(f"x0: {np.array2string(_x0, separator=", ", sign="+")}, Parameter: {param}")
            case "e":
                erase_lines()
                print("Erased.")
            case "a":
                param.increment(_pk[0], -inc_param_step)
            case "d":
                param.increment(_pk[0], inc_param_step)
            case "w":
                param.increment(_pk[1], inc_param_step)
            case "s":
                param.increment(_pk[1], -inc_param_step)
            case " ":
                is_paused = not is_paused
                if is_paused:
                    ani.event_source.stop()
                    print("Paused.")
                else:
                    ani.event_source.start()
                    print("Resumed.")
            case "q":
                print("Quit.")
                sys.exit()
            case _:
                print("No action for the key.")

        # If updated parameter
        if event.key in ("a", "d", "w", "s"):
            erase_lines()
            print(f"Parameter updated: {param}")

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
    ani = FuncAnimation(fig, update, frames=itr_max, interval=50)
    plt.show()


def main():
    try:
        with open(sys.argv[1], "r") as f:
            data = json.load(f)
        x0 = np.array(data.get("x0", [0, 0]))
        param = Parameter(**data.get("parameters", {}))
        mode = data.get("traj_mode", "animation")

    except IndexError:
        raise IndexError("Usage: python traj.py [data.json]")

    if mode == "animation":
        ax_config = {
            "xlim": (-1, 1),
            "ylim": (-1, 1),
            "aspect": "equal",
            "xlabel": "x",
            "ylabel": "y",
        } | data.get("ax_config", {})

        inc_param_keys = data.get("inc_param_keys", ["B", "B0"])

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set(**ax_config)
        ax.grid()

        traj_animation(fig, ax, x0, param, inc_param_keys)
    elif mode == "dump":
        pass
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
