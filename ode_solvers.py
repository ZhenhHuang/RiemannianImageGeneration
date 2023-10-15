import torch
from geoopt.manifolds.stereographic.math import expmap, logmap, project, dist
from torch.autograd.functional import jacobian


def ode_step(xt, vt, dt, kappa):
    if kappa != 0:
        return expmap(xt, vt * dt, k=kappa)
    else:
        return xt + vt * dt


@torch.no_grad()
def integrator(ode_func, x0, t, kappa):
    """
    Input:
        ode_func: f(t, x) -> (1, B, C, H, W)
        x0: (B, C, H, W)
        t: (T)
        kappa: torch.tensor()
    Return:
        xts: (T, B, C, H, W)
        vts: (T, B, C, H, W)
    """
    t0s = t[:-1]
    t1s = t[1:]
    vts = []
    xts = [x0.unsqueeze(0)]

    xt = x0.unsqueeze(0)
    for t0, t1 in zip(t0s, t1s):
        dt = t1 - t0
        vt = ode_func(t0, xt)
        xt = ode_step(xt, vt, dt, kappa)
        if kappa != 0:
            xt = project(xt, k=kappa)
        vts.append(vt)
        xts.append(xt)
    vts.append(ode_func(t1, xt))
    return torch.concat(xts), torch.concat(vts)


@torch.no_grad()
def solve_geodesic(t, x0, x1, kappa):
    """
    Solve geodesic On Stereographic model.
    :param t:  (T, )
    :param x0: (B, C, H, W)
    :param x1: (B, C, H, W)
    :param kappa: torch.tensor([])
    :return: [Xt, Vt] with shape (T, B, C, H, W)
    """
    x0 = x0.unsqueeze(0)
    x1 = x1.unsqueeze(0)
    t = t[:, None, None, None, None]

    def geodesic(time):
        if kappa == 0:
            return x0 + t * (x1 - x0)
        else:
            return expmap(x0, time * logmap(x0, x1, k=kappa), k=kappa)
    xt = geodesic(t)
    dxt_dt = jacobian(geodesic, t).sum(-1)    # (T, B, C, H, W)
    return xt, dxt_dt