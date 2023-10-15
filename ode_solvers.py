import torch
from geoopt.manifolds.stereographic.math import expmap, project, dist
from torch.func import vjp, vmap


def ode_step(xt, vt, dt, kappa):
    if kappa != 0:
        return expmap(xt, vt * dt, k=kappa)
    else:
        return xt + vt * dt


@torch.no_grad()
def integrator(ode_func, x0, t, kappa):
    """
    Input:
        ode_func: f(t, x)
        x0: (N, D)
        t: (T)
        kappa: torch.tensor()
    Return:
        xts: (T, N, D)
        vts: (T, N, D)
    """
    t0s = t[:-1]
    t1s = t[1:]
    vts = []
    xts = [x0]

    xt = x0
    for t0, t1 in zip(t0s, t1s):
        dt = t1 - t0
        vt = ode_func(t0, xt)
        xt = ode_step(xt, vt, dt, kappa)
        if kappa != 0:
            xt = project(xt, k=kappa)
        vts.append(vt)
        xts.append(xt)
    vts.append(ode_func(t1, xt))
    return torch.stack(xts), torch.stack(vts)


@torch.no_grad()
def solve_geodesic(x0, x1, t, kappa):
    orig_dist = dist(x0, x1, k=kappa)

    def odefunc(t, x):
        del t

        d, vjp_fn = vjp(lambda x: dist(x, x1, k=kappa), x)
        dgradx = vjp_fn(torch.ones_like(d))[0]

        dx = (
                -orig_dist[..., None]
                * dgradx
                / torch.linalg.norm(dgradx, dim=-1, keepdim=True)
                .pow(2)
                .clamp(min=1e-20)
        )
        return dx

    return integrator(odefunc, x0, t, kappa)