import sys, json

from collections.abc import Callable
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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

    for i in range(2):
        if fix_result["abs_eig"][i] > 1:
            unstable_vec = fix_result["vec"][:, i]
            u_itr_cnt = 2 if np.sign(fix_result["eig"][i]) == -1 else 1
        else:
            stable_vec = fix_result["vec"][:, i]
            s_itr_cnt = 2 if np.sign(fix_result["eig"][i]) == -1 else 1

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

    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
