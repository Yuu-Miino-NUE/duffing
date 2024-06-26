"""Module to calculate and draw manifolds of a fixed point.

Examples
--------
This module works with 4 modes: `animation`, `dump`, `search`, and `draw`.

1. Animation mode:
    This mode calculates and draws manifolds of a fixed point with animation.
    Run with the following command:

    .. code-block:: bash

        python manifold.py [data.json]


    where `data.json` is a JSON file that contains the following

    .. code-block:: json

        {
            "xfix": [
                -0.95432111,
                0.19089924
            ],
            "parameters": {
                "k": 0.05,
                "B0": 0,
                "B": 0.3
            },
            "period": 1,
            "manifold_mode": "animation"
        }

    The ``matplotlib`` window will be opened and the animation will be shown.

2. Dump mode:
    This mode calculates and dumps manifold data to CSV files.
    Run with the following command:

    .. code-block:: bash

        python manifold.py [data.json]

    where `data.json` is a JSON file that contains the following

    .. code-block:: json

        {
            "xfix": [
                -0.95432111,
                0.19089924
            ],
            "parameters": {
                "k": 0.05,
                "B0": 0,
                "B": 0.3
            },
            "period": 1,
            "manifold_mode": "dump"
        }

    The manifold data will be dumped to CSV files with the same name but with the suffix `_unstable_mani{i}.csv` and `_stable_mani{i}.csv`.

3. Search mode:
    This mode searches for a homoclinic point.
    Before running this mode, you need to run the dump mode first.
    Run with the following command:

    .. code-block:: bash

        python manifold.py [data.json]

    where `data.json` is a JSON file that contains the following

    .. code-block:: json

        {
            "xfix": [
                -0.95432111,
                0.19089924
            ],
            "parameters": {
                "k": 0.05,
                "B0": 0,
                "B": 0.3
            },
            "period": 1,
            "manifold_mode": "search"
        }

    The ``matplotlib`` window will be opened and you can search for the closest homoclinic point.
    Using the arrow keys, you can move the initial points of the unstable and stable manifolds.
    See the key bindings section of :func:`manifold.manifold_animation` for more details.

4. Draw mode:
    This mode draws manifolds with the dumped data.
    Before running this mode, you need to run the dump mode first.
    Run with the following command:

    .. code-block:: bash

        python manifold.py [data.json]

    where `data.json` is a JSON file that contains the following

    .. code-block:: json

        {
            "xfix": [
                -0.95432111,
                0.19089924
            ],
            "parameters": {
                "k": 0.05,
                "B0": 0,
                "B": 0.3
            },
            "period": 1,
            "manifold_mode": "draw"
        }

    The ``matplotlib`` window will be opened and the manifolds will be drawn.


.. note::
    Parameters for subfunctions can be set in the JSON file like the following:

    .. code-block:: json

        {
            "manifold_config": {
                "allowed_distance": {
                    "u": 1e-2,
                    "s": 1e-2
                },
                "curvature_threshold": {
                    "u": 0.1,
                    "s": 0.1
                },
                "m_max": {
                    "u": 100,
                    "s": 100
                },
                "init_resolution": {
                    "u": 100,
                    "s": 100
                },
                "vec_size": {
                    "u": 1e-4,
                    "s": 1e-4
                },
                "max_frames": 3,
                "itr_cnt": {
                    "u": 5,
                    "s": 5
                }
            }
        }

    where "u" and "s" are for the unstable and stable manifolds, respectively.

"""

import sys, json

from collections.abc import Callable
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy
import numpy as np

from system import poincare_map, Parameter
from fix import fix


def calc_curvature(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    """Calculate curvature of the curve.

    Parameters
    ----------
    x : numpy.ndarray
        x-coordinates of the curve.
    y : numpy.ndarray
        y-coordinates of the curve.

    Returns
    -------
    numpy.ndarray
        Curvatures of the curve for each point.
    """
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    velocity = np.array([dx_dt, dy_dt]).T

    ds_dt = np.sqrt(dx_dt**2 + dy_dt**2)
    tangent = np.array([velocity[i] / ds_dt[i] for i in range(len(ds_dt))])

    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]

    d_tangent_x_dt = np.gradient(tangent_x)
    d_tangent_y_dt = np.gradient(tangent_y)

    d_tangent_dt = np.array([d_tangent_x_dt, d_tangent_y_dt]).T

    curvature = np.linalg.norm(d_tangent_dt, axis=1) / ds_dt
    return curvature


def remove_similar(x: numpy.ndarray, tol: float = 1e-2) -> numpy.ndarray:
    """Remove similar points in the array.

    Parameters
    ----------
    x : numpy.ndarray
        Array of points.
    tol : float, optional
        Tolerance of the distance to determine similarity, by default 1e-2.

    Returns
    -------
    numpy.ndarray
        Array of unique points.
    """
    x = np.array(x)
    x_unique = [x[0]]
    for i in range(1, len(x)):
        if np.linalg.norm(x[i] - x_unique[-1]) > tol:
            x_unique.append(x[i])
    return np.array(x_unique)


def calculate_next_manifold(
    domain: numpy.ndarray,
    func: Callable[[numpy.ndarray], numpy.ndarray],
    allowed_distance: float = 1e-2,
    curvature_threshold: float = 0.1,
    m_max: int = 100,
    final: bool = False,
) -> numpy.ndarray:
    """Calculate next manifold from a given domain.

    Parameters
    ----------
    domain : numpy.ndarray
        Domain of the calculation.
    func : Callable[[numpy.ndarray], numpy.ndarray]
        Function to calculate the next point.
    allowed_distance : float, optional
        Allowed distance between points, by default 1e-2. In the case of the distance between points is less than this value and the curvature is less than the threshold, the calculation stops.
    curvature_threshold : float, optional
        Threshold of the curvature to determine the end of the calculation, by default 0.1.
    m_max : int, optional
        Maximum number of domain divider, by default 100.
    final : bool, optional
        Flag to determine the final calculation, by default False.

    Returns
    -------
    numpy.ndarray
        Calculated manifold.
    """
    image = np.array([func(x) for x in domain])
    curvature = calc_curvature(image[:, 0], image[:, 1])[:-1]
    tf_table = [
        tf[0] or tf[1]
        for tf in zip(
            np.linalg.norm(image[1:] - image[0:-1], axis=1) < allowed_distance,
            curvature < curvature_threshold,
        )
    ]

    if np.all(tf_table) or final:
        image = remove_similar(image, tol=allowed_distance * 1e-2)
        return image
    else:
        new_image = np.empty((0, 2))
        _m = 1
        for i in range(len(image) - 1):
            new_image = np.vstack((new_image, image[i]))

            if not tf_table[i]:
                _m = np.ceil(np.linalg.norm(image[i + 1] - image[i]) / allowed_distance)

                new_image = np.vstack(
                    (
                        new_image,
                        calculate_next_manifold(
                            np.array(
                                [
                                    domain[i]
                                    + (domain[i + 1] - domain[i]) * (j + 1) / _m
                                    for j in range(int(_m))
                                ]
                            ),
                            func,
                            allowed_distance,
                            m_max,
                            final=(_m > m_max),
                        ),
                    )
                )

        new_image = np.vstack((new_image, image[-1]))
        new_image = remove_similar(new_image, tol=allowed_distance * 1e-2)
        return new_image


def manifold_animation(
    fig: Figure,
    ax: Axes,
    xfix: numpy.ndarray,
    domain: dict[str, numpy.ndarray],
    func: dict[str, Callable[[numpy.ndarray], numpy.ndarray]],
    max_frames: int = 3,
    allowed_distance: dict[str, float] = {"u": 1e-2, "s": 1e-2},
    curvature_threshold: dict[str, float] = {"u": 0.1, "s": 0.1},
    m_max: dict[str, int] = {"u": 100, "s": 100},
):
    """Animation to draw manifolds.

    Parameters
    ----------
    fig : Figure
        Figure object.
    ax : Axes
        Axes object.
    xfix : numpy.ndarray
        Fixed point.
    domain : dict[str, numpy.ndarray]
        Domain of the unstable manifold and the stable manifold.
    func : dict[str, Callable[[numpy.ndarray], numpy.ndarray]]
        Function to calculate the next point of the unstable manifold and the stable manifold.
    max_frames : int, optional
        Maximum number of frames of animation, by default 3.
    allowed_distance : dict[str, float], optional
        Allowed distance between points for each minifold, by default 1e-2 for both.
    curvature_threshold : dict[str, float], optional
        Threshold of the curvature to determine the end of the calculation, by default 0.1 for both manifolds.
    m_max : dict[str, int], optional
        Maximum number of domain divider, by default 100 for both manifolds.


    Key Bindings
    ------------
    ============ ======
    Key          Action
    ============ ======
    :kbd:`left`  Move the unstable manifold to the left.
    :kbd:`right` Move the unstable manifold to the right.
    :kbd:`a`     Move the unstable manifold to the left in small steps.
    :kbd:`d`     Move the unstable manifold to the right in small steps.
    :kbd:`up`    Move the stable manifold up.
    :kbd:`down`  Move the stable manifold down.
    :kbd:`w`     Move the stable manifold up in small steps.
    :kbd:`s`     Move the stable manifold down in small steps.
    :kbd:`0`     Reset the moving small step factor to 1/10.
    :kbd:`-`     Decrease the moving small step factor by 1/10.
    :kbd:`=`     Increase the moving small step factor by 10.
    :kbd:`p`     Print the current points of the fixed point, the unstable manifold, and the stable manifold.
    :kbd:`q`     Close the window.
    ============ ======

    """
    ax.plot(xfix[0], xfix[1], "ko")
    colors = {"u": "r", "s": "b"}

    for us in ["u", "s"]:
        ax.plot(domain[us][:, 0], domain[us][:, 1], "-", color=colors[us])

    _domain_u = domain["u"]
    _domain_s = domain["s"]

    def update(i: int):
        nonlocal _domain_u, _domain_s
        new_mani = {}
        __domain = {"u": _domain_u, "s": _domain_s}
        for us in ["u", "s"]:
            __domain[us] = calculate_next_manifold(
                __domain[us],
                func[us],
                allowed_distance=allowed_distance[us],
                curvature_threshold=curvature_threshold[us],
                m_max=m_max[us],
            )
            new_mani[us] = ax.plot(
                __domain[us][:, 0], __domain[us][:, 1], "-", color=colors[us]
            )

        _domain_u = __domain["u"]
        _domain_s = __domain["s"]
        print(f"Frame {i + 1}/{max_frames}")
        return new_mani["s"] + new_mani["u"]

    ani = FuncAnimation(fig, update, frames=max_frames, interval=50, repeat=False)
    plt.show()


def dump_manifold(
    domain: numpy.ndarray,
    func: Callable[[numpy.ndarray], numpy.ndarray],
    itr_cnt: int = 3,
    **kwargs,
):
    """Dump manifold data calculated with a given func, itr_cnt, and domain.

    Parameters
    ----------
    domain : numpy.ndarray
        Domain of the manifold.
    func : Callable[[numpy.ndarray], numpy.ndarray]
        Function to calculate the next point of the manifold.
    itr_cnt : int, optional
        Iteration count of the calculation, by default 3.
    kwargs : dict
        Keyword arguments for calculate_next_manifold.

    Returns
    -------
    numpy.ndarray
        Dumped manifold data.
    """
    _domain = domain.copy()
    ret = _domain.copy()
    for _ in range(itr_cnt):
        _domain = calculate_next_manifold(_domain, func, **kwargs)
        ret = np.vstack((ret, _domain))
    return ret


def append_info_for_mani(mani: numpy.ndarray) -> numpy.ndarray:
    """Append curvature and distance information to the manifold data.

    Parameters
    ----------
    mani : numpy.ndarray
        Manifold data.

    Returns
    -------
    numpy.ndarray
        Manifold data with curvature and distance information.
    """
    curve = calc_curvature(mani[:, 0], mani[:, 1])
    dist = np.insert(np.linalg.norm(mani[1:] - mani[0:-1], axis=1), -1, 0)
    return np.hstack((mani, curve.reshape(-1, 1), dist.reshape(-1, 1)))


def draw_manifold(
    ax: Axes,
    xfix: np.ndarray,
    um_data: list[numpy.ndarray],
    sm_data: list[numpy.ndarray],
):
    """Draw manifolds.

    Parameters
    ----------
    ax : Axes
        Axes object.
    xfix : numpy.ndarray
        Fixed point.
    um_data : list[numpy.ndarray]
        Unstable manifold data.
    sm_data : list[numpy.ndarray]
        Stable manifold data.
    """
    for i in range(2):
        ax.plot(um_data[i][:, 0], um_data[i][:, 1], "r-")
        ax.plot(sm_data[i][:, 0], sm_data[i][:, 1], "b-")
    ax.legend(["Unstable manifold", "Stable manifold"])
    ax.plot(xfix[0], xfix[1], "ko")


def setup_finder(
    fig: Figure,
    ax: Axes,
    xfix: numpy.ndarray,
    funcs: dict[str, Callable[[numpy.ndarray], numpy.ndarray]],
    vecs: dict[str, numpy.ndarray],
    itr_cnts: dict[str, int] = {"u": 5, "s": 5},
    bases: dict[str, int] = {"u": 1, "s": 1},
):
    """Setup finder for the closest homoclinic point.

    Parameters
    ----------
    fig : Figure
        Figure object.
    ax : Axes
        Axes object.
    xfix : numpy.ndarray
        Fixed point.
    funcs : dict[str, Callable[[numpy.ndarray], numpy.ndarray]]
        Function to calculate the next point of the unstable and stable manifolds.
    u_evec : dict[str, numpy.ndarray]
        Unstable and stable eigenvectors.
    itr_cnts : dict[str, int], optional
        Iteration counts of the calculation, by default 5 for both manifolds.
    bases : dict[str, int], optional
        Multiplier of the unstable and stable eigenvector, by default 1 for both.
    """

    def plot_points(
        x0: numpy.ndarray,
        func: Callable[[numpy.ndarray], numpy.ndarray],
        itr_cnt: int,
        color: str,
    ) -> Line2D:
        x = x0
        points = np.zeros((itr_cnt, 2))
        for i in range(itr_cnt):
            x = func(x)
            points[i] = x
        return ax.plot(points[:, 0], points[:, 1], f"{color}o", markersize=5)[0]

    bias_base = 1 / 10
    bias_u = 0
    bias_s = 0
    base_u = bases["u"]
    base_s = bases["s"]

    colors = {"u": "r", "s": "b"}

    args = {
        k: {
            "x0": xfix + bases[k] * vecs[k],
            "func": funcs[k],
            "color": colors[k],
        }
        for k in ["u", "s"]
    }

    points: dict[str, Line2D] = {}
    for k in args.keys():
        points[k] = plot_points(**args[k], itr_cnt=itr_cnts[k])

    def on_press(event):
        nonlocal points, args, base_u, base_s, bias_base, bias_u, bias_s
        print(
            f"press {event.key}" + ("\t" if len(event.key) != 1 else "\t\t") + "| ",
            end="",
        )
        match event.key:
            case "right" | "left" | "d" | "a":
                if event.key in ["right", "left"]:
                    base_u += 1 if event.key == "right" else -1
                else:
                    bias_u += 1 if event.key == "d" else -1

                points["u"].remove()

                xu = xfix + (base_u + bias_base * bias_u) * vecs["u"]
                args["u"]["x0"] = xu
                points["u"] = plot_points(**args["u"], itr_cnt=itr_cnts["u"])

                fig.canvas.draw()
                print(
                    f"{itr_cnts['u']} points in Unstable manifold:\tx_u(0) = {xu}\t|",
                    f"err={np.linalg.norm(xu - xfix):.3e}",
                )
            case "up" | "down" | "w" | "s":
                if event.key in ["up", "down"]:
                    base_s += 1 if event.key == "up" else -1
                else:
                    bias_s += 1 if event.key == "w" else -1

                points["s"].remove()

                xs = xfix + (base_s + bias_base * bias_s) * vecs["s"]
                args["s"]["x0"] = xs
                points["s"] = plot_points(**args["s"], itr_cnt=itr_cnts["s"])

                fig.canvas.draw()
                print(
                    f"{itr_cnts['s']} points in Stable manifold:\tx_s(0) = {xs}\t|",
                    f"err={np.linalg.norm(xs - xfix):.3e}",
                )
            case "0" | "-" | "=":
                if event.key == "0":
                    bias_base = 1 / 10
                elif event.key == "-":
                    bias_base /= 10
                elif event.key == "=":
                    bias_base *= 10

                print(f"Factor for moving step: {bias_base}")
            case "p":
                xu = xfix + (base_u + bias_base * bias_u) * vecs["u"]
                xs = xfix + (base_s + bias_base * bias_s) * vecs["s"]
                print(
                    f'"x0": {xfix.tolist()}, "xu": {xu.tolist()}, "xs": {xs.tolist()}'
                )
            case "q":
                plt.close()
                print("Closed.")
            case _:
                print("Invalid key.")
                pass

    def on_click(event):
        print(f"click: {event.xdata:+.3f}, {event.ydata:+.3f} | ", end="")
        x = {}
        for k in points.keys():
            min_dist = None
            min_i = 1
            for i, _x in enumerate(np.squeeze(np.array(points[k].get_data())).T):
                if i == 0:
                    _x0 = _x
                    _xi = _x
                new_dist = np.linalg.norm(np.array([event.xdata, event.ydata]) - _x)
                if min_dist is None or new_dist < min_dist:
                    min_dist = new_dist
                    min_i = i + 1
                    _xi = _x

            x[k] = {
                "x0": _x0,
                "xi": _xi,
                "i": min_i,
            }
        dist = np.linalg.norm(x["u"]["xi"] - x["s"]["xi"])
        outlist = [(us["xi"].tolist(), us["i"]) for us in x.values()]
        print(f"distance={dist:.3e} | Closest Points: {outlist}")

    # Connect event handlers
    if (manager := fig.canvas.manager) is not None and (
        hid := manager.key_press_handler_id
    ) is not None:
        fig.canvas.mpl_disconnect(hid)  # Disconnect default key press handler
    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("button_press_event", on_click)


def _prepare_by_fix(xfix0: numpy.ndarray, param: Parameter, period):
    # Calculate fixed point and
    fix_result = fix(xfix0, param, period=period)
    if not fix_result.success:
        raise ValueError(fix_result.message)

    # Check fixed point and eigenspaces
    if (xfix := fix_result.xfix) is None:
        raise ValueError("Failed to find fixed point")
    if fix_result.u_edim != 1 or fix_result.s_edim != 1:
        raise ValueError("Invalid dimension of eigenspace")
    if (u_evec := fix_result.u_evec) is None:
        raise ValueError("Failed to find unstable eigenvector")
    if (s_evec := fix_result.s_evec) is None:
        raise ValueError("Failed to find stable eigenvector")
    if (u_eig := fix_result.u_eig) is None:
        raise ValueError("Failed to find unstable eigenvalue")
    if (s_eig := fix_result.s_eig) is None:
        raise ValueError("Failed to find stable eigenvalue")

    # Setup variables
    u_itr_cnt = 2 if np.sign(u_eig[0]) == -1 else 1
    s_itr_cnt = 2 if np.sign(s_eig[0]) == -1 else 1

    return (xfix, u_evec[:, 0], s_evec[:, 0], u_itr_cnt, s_itr_cnt)


def _main():
    # Load JSON
    try:
        with open(sys.argv[1], "r") as f:
            data = json.load(f)
        xfix0 = np.array(data.get("xfix", [0, 0]))
        param = Parameter(**data.get("parameters", {}))
        period = data.get("period", 1)
        mode = data.get("manifold_mode", "animation")
    except IndexError:
        raise IndexError("Usage: python manifold.py [data.json]")
    except FileNotFoundError:
        raise FileNotFoundError(f"{sys.argv[1]} not found")

    xfix, u_evec, s_evec, u_itr_cnt, s_itr_cnt = _prepare_by_fix(xfix0, param, period)

    u_func = lambda x: poincare_map(
        x, param, itr_cnt=u_itr_cnt * period, calc_jac=True
    ).x
    s_func = lambda x: poincare_map(
        x, param, itr_cnt=s_itr_cnt * period, inverse=True, calc_jac=True
    ).x

    # Setup figure and axes
    if mode in ["animation", "search", "draw"]:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax_config = {
            "xlabel": "x",
            "ylabel": "y",
            "xlim": (-2, 2),
            "ylim": (-2, 2),
        }
        ax.set(**ax_config)
        ax.grid()

    # Main process
    config = data.get("manifold_config", {})
    if mode in ["animation", "dump"]:
        allowed_distance = config.get("allowed_distance", {"u": 1e-2, "s": 1e-2})
        curvature_threshold = config.get("curvature_threshold", {"u": 0.1, "s": 0.1})
        m_max = config.get("m_max", {"u": 100, "s": 100})
        init_resolution = config.get("init_resolution", {"u": 100, "s": 100})
        vec_size = config.get("vec_size", {"u": 1e-4, "s": 1e-4})

        if mode == "animation":
            max_frames = config.get("max_frames", 3)

            # Setup eigenspaces
            eigspace = {}
            vecs = {"u": u_evec, "s": s_evec}
            funcs = {"u": u_func, "s": s_func}
            for us in ["u", "s"]:
                eigspace[us] = [
                    xfix + eps * vecs[us] for eps in (-vec_size[us], vec_size[us])
                ]
                eigspace[us] = np.linspace(
                    funcs[us](eigspace[us][0]),
                    funcs[us](eigspace[us][1]),
                    init_resolution[us],
                    endpoint=False,
                )

            # Draw manifolds
            manifold_animation(
                fig=fig,
                ax=ax,
                xfix=xfix,
                domain=eigspace,
                func=funcs,
                max_frames=max_frames,
                allowed_distance=allowed_distance,
                curvature_threshold=curvature_threshold,
            )
        else:  # mode == "dump"
            itr_cnt = config.get("itr_cnt", {"u": 5, "s": 5})
            manis = {"u": [], "s": []}
            vecs = {"u": u_evec, "s": s_evec}
            funcs = {"u": u_func, "s": s_func}
            labels = {"u": "unstable", "s": "stable"}

            for s in [1, -1]:
                eigspace = {}
                _mani = {}
                for us in ["u", "s"]:
                    eigspace[us] = [xfix + s * vec_size[us] * vecs[us]]
                    eigspace[us] = np.linspace(
                        eigspace[us][0],
                        funcs[us](eigspace[us][0]),
                        init_resolution[us],
                        endpoint=False,
                    )
                    print(
                        "Calculating manifold for", labels[us], "with", s, "direction"
                    )
                    _mani[us] = dump_manifold(
                        eigspace[us],
                        funcs[us],
                        itr_cnt=itr_cnt[us],
                        allowed_distance=allowed_distance[us],
                        curvature_threshold=curvature_threshold[us],
                        m_max=m_max[us],
                    )
                    _mani[us] = append_info_for_mani(_mani[us])
                    manis[us].append(_mani[us])

            for i in range(2):
                for us in ["u", "s"]:
                    with open(
                        sys.argv[1].replace(".json", f"_{labels[us]}_mani{i}.csv"), "w"
                    ) as f:
                        np.savetxt(f, manis[us][i], delimiter=",")

    elif mode == "search" or mode == "draw":
        try:
            um_data = [
                np.loadtxt(
                    sys.argv[1].replace(".json", f"_unstable_mani{i}.csv"),
                    delimiter=",",
                )
                for i in range(2)
            ]
            sm_data = [
                np.loadtxt(
                    sys.argv[1].replace(".json", f"_stable_mani{i}.csv"),
                    delimiter=",",
                )
                for i in range(2)
            ]
        except FileNotFoundError:
            raise FileNotFoundError(
                "Manifold data not found. Run with 'dump' mode first."
            )

        draw_manifold(ax, xfix, um_data, sm_data)

        if mode == "search":
            eps = config.get("vec_size", {"u": 1e-4, "s": 1e-4})
            itr_cnts = config.get("itr_cnt", {"u": 5, "s": 5})

            funcs = {"u": u_func, "s": s_func}
            vecs = {"u": u_evec * eps["u"], "s": s_evec * eps["s"]}

            setup_finder(fig, ax, xfix, funcs, vecs, itr_cnts)

        plt.show()

    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    _main()
