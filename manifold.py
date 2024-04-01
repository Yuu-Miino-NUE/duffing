import sys, json

from collections.abc import Callable
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from duffing import poincare_map, Parameter
from fix import fix

from curvature import calc_curvature


def remove_similar(x: np.ndarray, tol: float = 1e-2) -> np.ndarray:
    x = np.array(x)
    x_unique = [x[0]]
    for i in range(1, len(x)):
        if np.linalg.norm(x[i] - x_unique[-1]) > tol:
            x_unique.append(x[i])
    return np.array(x_unique)


def calculate_next_manifold(
    domain: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
    allowed_distance: float = 1e-2,
    curvature_threshold: float = 0.1,
    m_max: int = 100,
    final: bool = False,
) -> np.ndarray:
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
    x_fix: np.ndarray,
    domain_u: np.ndarray,
    func_u: Callable[[np.ndarray], np.ndarray],
    domain_s: np.ndarray,
    func_s: Callable[[np.ndarray], np.ndarray],
    allowed_distance: float = 1e-2,
    max_frames: int = 3,
):
    ax.plot(x_fix[0], x_fix[1], "ko")

    ax.plot(domain_s[:, 0], domain_s[:, 1], "b-")
    ax.plot(domain_u[:, 0], domain_u[:, 1], "r-")

    def update(i: int):
        nonlocal domain_s, domain_u
        domain_s = calculate_next_manifold(
            domain_s, func_s, allowed_distance=allowed_distance
        )
        new_mani_s = ax.plot(domain_s[:, 0], domain_s[:, 1], "b-")
        domain_u = calculate_next_manifold(
            domain_u, func_u, allowed_distance=allowed_distance
        )
        new_mani_u = ax.plot(domain_u[:, 0], domain_u[:, 1], "r-")
        print(f"Iteration count: {i + 1}")
        return new_mani_s + new_mani_u

    ani = FuncAnimation(fig, update, frames=max_frames, interval=50, repeat=False)
    plt.show()


def dump_manifold(
    domain: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
    itr_cnt: int = 3,
    allowed_distance: float = 1e-2,
):
    _domain = domain.copy()
    ret = _domain.copy()
    for _ in range(itr_cnt):
        _domain = calculate_next_manifold(
            _domain, func, allowed_distance=allowed_distance
        )
        ret = np.vstack((ret, _domain))
    return ret


def append_info_for_mani(mani: np.ndarray) -> np.ndarray:
    curve = calc_curvature(mani[:, 0], mani[:, 1])
    dist = np.insert(np.linalg.norm(mani[1:] - mani[0:-1], axis=1), -1, 0)
    return np.hstack((mani, curve.reshape(-1, 1), dist.reshape(-1, 1)))


def draw_manifold(ax: Axes, um_data: np.ndarray, sm_data: np.ndarray):
    ax.plot(um_data[:, 0], um_data[:, 1], "r-", label="Unstable manifold")
    ax.plot(sm_data[:, 0], sm_data[:, 1], "b-", label="Stable manifold")
    ax.legend()


def setup_finder(
    fig: Figure,
    ax: Axes,
    x_fix: np.ndarray,
    func_u: Callable[[np.ndarray], np.ndarray],
    func_s: Callable[[np.ndarray], np.ndarray],
    unstable_vec: np.ndarray,
    stable_vec: np.ndarray,
    itr_cnt: int = 5,
    base_u: int = 1,
    base_s: int = 1,
):
    def plot_points(
        x0: np.ndarray,
        func: Callable[[np.ndarray], np.ndarray],
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

    args = {
        "u": {
            "x0": x_fix + base_u * unstable_vec,
            "func": func_u,
            "color": "r",
        },
        "s": {
            "x0": x_fix + base_s * stable_vec,
            "func": func_s,
            "color": "b",
        },
    }

    points: dict[str, Line2D] = {}
    for k in args.keys():
        points[k] = plot_points(**args[k], itr_cnt=itr_cnt)

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

                xu = x_fix + (base_u + bias_base * bias_u) * unstable_vec
                args["u"]["x0"] = xu
                points["u"] = plot_points(**args["u"], itr_cnt=itr_cnt)

                fig.canvas.draw()
                print(
                    f"{itr_cnt} points in Unstable manifold:\tx_u(0) = {xu}\t|",
                    f"err={np.linalg.norm(xu - x_fix):.3e}",
                )
            case "up" | "down" | "w" | "s":
                if event.key in ["up", "down"]:
                    base_s += 1 if event.key == "up" else -1
                else:
                    bias_s += 1 if event.key == "w" else -1

                points["s"].remove()

                xs = x_fix + (base_s + bias_base * bias_s) * stable_vec
                args["s"]["x0"] = xs
                points["s"] = plot_points(**args["s"], itr_cnt=itr_cnt)

                fig.canvas.draw()
                print(
                    f"{itr_cnt} points in Stable manifold:\tx_s(0) = {xs}\t|",
                    f"err={np.linalg.norm(xs - x_fix):.3e}",
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
                xu = x_fix + (base_u + bias_base * bias_u) * unstable_vec
                xs = x_fix + (base_s + bias_base * bias_s) * stable_vec
                print(
                    f'"x0": {x_fix.tolist()}, "xu": {xu.tolist()}, "xs": {xs.tolist()}'
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


def main():
    try:
        with open(sys.argv[1], "r") as f:
            data = json.load(f)
        x0 = np.array(data.get("x0", [0, 0]))
        param = Parameter(**data.get("parameters", {}))
        period = data.get("period", 1)
        mode = data.get("manifold_mode", "animation")
    except IndexError:
        raise IndexError("Usage: python manifold.py [data.json]")
    except FileNotFoundError:
        raise FileNotFoundError(f"{sys.argv[1]} not found")

    fix_result = fix(x0, param, period=period)
    print(fix_result)

    x_fix = fix_result["x"]
    unstable_vec = None
    stable_vec = None

    for i in range(2):
        if fix_result["abs_eig"][i] > 1:
            unstable_vec = fix_result["vec"][:, i]
            u_itr_cnt = 2 if np.sign(fix_result["eig"][i]) == -1 else 1
        else:
            stable_vec = fix_result["vec"][:, i]
            s_itr_cnt = 2 if np.sign(fix_result["eig"][i]) == -1 else 1

    if unstable_vec is None or stable_vec is None:
        raise ValueError("Eigenvectors are not found")

    unstable_func = lambda x: poincare_map(x, param, itr_cnt=u_itr_cnt)["x"]
    stable_func = lambda x: poincare_map(x, param, itr_cnt=s_itr_cnt, inverse=True)["x"]

    if mode == "animation":
        unstable_x = [x_fix + eps * unstable_vec for eps in (-1e-2, 1e-2)]
        unstable_eig_space = np.linspace(
            unstable_func(unstable_x[0]),
            unstable_func(unstable_x[1]),
            10,
            endpoint=False,
        )

        stable_x = [x_fix + eps * stable_vec for eps in (-1e-2, 1e-2)]
        stable_eig_space = np.linspace(
            stable_func(stable_x[0]), stable_func(stable_x[1]), 10, endpoint=False
        )

        fig, ax = plt.subplots(figsize=(8, 7))
        ax_config = {
            "xlabel": "x",
            "ylabel": "y",
            "xlim": (-2, 2),
            "ylim": (-2, 2),
        }
        ax.set(**ax_config)
        ax.grid()

        # manifold_animation(fig, ax, x_fix, unstable_eig_space, unstable_func)
        manifold_animation(
            fig=fig,
            ax=ax,
            x_fix=x_fix,
            domain_s=stable_eig_space,
            func_s=stable_func,
            domain_u=unstable_eig_space,
            func_u=unstable_func,
            max_frames=3,
            allowed_distance=1e-2,
        )
    elif mode == "dump":
        unstable_x = x_fix + 1e-2 * unstable_vec
        unstable_eig_space = np.linspace(
            unstable_x, unstable_func(unstable_x), 10, endpoint=False
        )

        stable_x = x_fix - 1e-2 * stable_vec
        stable_eig_space = np.linspace(
            stable_x, stable_func(stable_x), 10, endpoint=False
        )

        unstable_mani = dump_manifold(
            unstable_eig_space, unstable_func, itr_cnt=4, allowed_distance=1e-2
        )
        unstable_mani = append_info_for_mani(unstable_mani)

        stable_mani = dump_manifold(
            stable_eig_space, stable_func, itr_cnt=4, allowed_distance=1e-2
        )
        stable_mani = append_info_for_mani(stable_mani)

        with open(sys.argv[1].replace(".json", "_unstable_mani.csv"), "w") as f:
            np.savetxt(f, unstable_mani, delimiter=",")
        with open(sys.argv[1].replace(".json", "_stable_mani.csv"), "w") as f:
            np.savetxt(f, stable_mani, delimiter=",")

    elif mode == "search":
        eps_u = 1e-4
        eps_s = 1e-4

        fig, ax = plt.subplots(figsize=(8, 7))
        ax_config = {
            "xlabel": "x",
            "ylabel": "y",
            "xlim": (-2, 2),
            "ylim": (-2, 2),
        }
        ax.set(**ax_config)
        ax.grid()

        try:
            um_data = np.loadtxt(
                sys.argv[1].replace(".json", "_unstable_mani.csv"), delimiter=","
            )
            sm_data = np.loadtxt(
                sys.argv[1].replace(".json", "_stable_mani.csv"), delimiter=","
            )
        except FileNotFoundError:
            raise FileNotFoundError("Manifold data not found")

        draw_manifold(ax, um_data, sm_data)

        unstable_vec *= eps_u
        stable_vec *= eps_s

        setup_finder(
            fig,
            ax,
            x_fix,
            unstable_func,
            stable_func,
            unstable_vec,
            stable_vec,
        )

        plt.show()

    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
