import numpy as np


def calc_curvature(x, y):
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
