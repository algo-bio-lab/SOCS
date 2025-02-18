import torch
import numpy as np



def log_ugw_sinkhorn_f(a, dx, b, dy, dc, alpha, init=None, eps=1.0,
                     rho=float("Inf"), rho2=None,
                     nits_plan=3000, tol_plan=1e-6,
                     nits_sinkhorn=3000, tol_sinkhorn=1e-6, alt=0,
                     two_outputs=False,print_per_iter=None):
    if rho2 is None:
        rho2 = rho
    # Initialize plan and local cost
    logpi = (init_plan(a, b, init=init) + 1e-30).log()
    logpi_prev = torch.zeros_like(logpi)
    up, vp = None, None
    losses = []
    for i in range(nits_plan):
        if print_per_iter is not None:
            if np.mod(i,print_per_iter)==0:
                print(i)
        logpi_prev = logpi.clone()
        lcost = compute_local_cost_f(logpi.exp(), a, dx, b, dy, dc, alpha, eps, rho, rho2)
        logmp = logpi.logsumexp(dim=(0, 1))
        up, vp, logpi = log_sinkhorn(
            lcost, up, vp, a, b, logmp.exp() + 1e-10, eps, rho, rho2,
            nits_sinkhorn, tol_sinkhorn,alt
        )
        if torch.any(torch.isnan(logpi)):
            raise Exception(
                f"Solver got NaN plan with params (eps, rho, rho2) "
                f" = {eps, rho, rho2}. Try increasing argument eps."
            )
        logpi = (
                0.5 * (logmp - logpi.logsumexp(dim=(0, 1)))
                + logpi
        )
        losses.append((logpi.exp()-logpi_prev.exp()).abs().max().item())
        if (logpi.exp() - logpi_prev.exp()).abs().max().item() < tol_plan:
            break

    if two_outputs:
        return logpi.exp(), logpi_prev.exp()
    return logpi.exp(),losses
    
def log_sinkhorn(lcost, f, g, a, b, mass, eps, rho, rho2, nits_sinkhorn,
                 tol_sinkhorn,alt):
    """
    Parameters
    ----------
    lcost: torch.Tensor of size [Batch, size_X, size_Y]
    Local cost depending on the current plan.

    f: torch.Tensor of size [Batch, size_X]
    First dual potential defined on X.

    g: torch.Tensor of size [Batch, size_Y]
    Second dual potential defined on Y.

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    mass: torch.Tensor of size [Batch]
    Mass of the current plan.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    nits_sinkhorn: int
    Maximum number of iterations to update Sinkhorn potentials in inner loop.

    tol_sinkhorn: float
    Tolerance on convergence of Sinkhorn potentials.

    Returns
    ----------
    u: torch.Tensor of size [Batch, size_X]
    First dual potential of Sinkhorn algorithm

    v: torch.Tensor of size [Batch, size_Y]
    Second dual potential of Sinkhorn algorithm

    logpi: torch.Tensor of size [Batch, size_X, size_Y]
    Optimal transport plan in log-space.
    """
    # Initialize potentials by finding best translation
    if f is None or g is None:
        f, g = torch.zeros_like(a), torch.zeros_like(b)
    f, g = log_translate_potential(f, g, lcost, a, b, mass, eps, rho, rho2)

    # perform Sinkhorn algorithm in LSE form
    s_x, s_y = aprox_softmin(lcost, a, b, mass, eps, rho, rho2)
    for j in range(nits_sinkhorn):
        if(alt==0):
            f_prev = f.clone()
            g = s_x(f)
            f = s_y(g)
            if (f - f_prev).abs().max().item() < tol_sinkhorn:
                break
        else:
            g_prev = g.clone()
            f = s_y(g)
            g = s_x(f)
            if (g - g_prev).abs().max().item() < tol_sinkhorn:
                break
    logpi = (
            (
                    (f[:, None] + g[None, :] - lcost)
                    / (mass * eps)
            )
            + a.log()[:, None]
            + b.log()[None, :]
    )
    return f, g, logpi

def compute_local_cost_f(pi, a, dx, b, dy, dc, alpha, eps, rho, rho2, complete_cost=True):
    """Compute the local cost by averaging the distortion with the current
    transport plan.

    Parameters
    ----------
    pi: torch.Tensor of size [Batch, size_X, size_Y]
    transport plan used to compute local cost

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    dx: torch.Tensor of size [Batch, size_X, size_X]
    Input metric of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    dy: torch.Tensor of size [Batch, size_Y, size_Y]
    Input metric of the second mm-space.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    complete_cost: bool
    If set to True, computes the full local cost, otherwise it computes the
    cross-part on (X,Y) to reduce computational complexity.

    Returns
    ----------
    lcost: torch.Tensor of size [Batch, size_X, size_Y]
    local cost depending on the current transport plan.
    """
    distxy = torch.einsum(
        "ij,kj->ik", dx, torch.einsum("kl,jl->kj", dy, pi)
    )
    kl_pi = torch.sum(
        pi * (pi / (a[:, None] * b[None, :]) + 1e-10).log()
    )
    if not complete_cost:
        return - 2 * distxy + eps * kl_pi

    mu, nu = torch.sum(pi, dim=1), torch.sum(pi, dim=0)
    distxx = torch.einsum("ij,j->i", dx ** 2, mu)
    distyy = torch.einsum("kl,l->k", dy ** 2, nu)

    lcost = (1-alpha)*(distxx[:, None] + distyy[None, :] - 2 * distxy) + eps * kl_pi
    #lcost = (1-alpha)*lcost

    if rho < float("Inf"):
        lcost = (
                lcost
                + rho
                * torch.sum(mu * (mu / a + 1e-10).log())
        )
    if rho2 < float("Inf"):
        lcost = (
                lcost
                + rho2
                * torch.sum(nu * (nu / b + 1e-10).log())
        )
    lcost = lcost + (alpha/2)*dc
    return lcost
    
def aprox_softmin(cost, a, b, mass, eps, rho, rho2):
    """Prepares functions which perform updates of the Sikhorn algorithm
    in logarithmic scale.

    Parameters
    ----------
    cost: torch.Tensor of size [Batch, size_X, size_Y]
    cost used in Sinkhorn iterations.

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    mass: torch.Tensor of size [Batch]
    Mass of the current plan.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    Returns
    ----------
    s_x: callable function
    Map outputing updates of potential from Y to X.

    s_y: callable function
    Map outputing updates of potential from X to Y.
    """

    tau = 1.0 / (1.0 + eps / rho)
    tau2 = 1.0 / (1.0 + eps / rho2)

    def s_y(g):
        return (
                -mass
                * tau2
                * eps
                * (
                        (g / (mass * eps) + b.log())[None, :]
                        - cost / (mass * eps)
                ).logsumexp(dim=1)
        )

    def s_x(f):
        return (
                -mass
                * tau
                * eps
                * (
                        (f / (mass * eps) + a.log())[:, None]
                        - cost / (mass * eps)
                ).logsumexp(dim=0)
        )

    return s_x, s_y
    
def log_translate_potential(u, v, lcost, a, b, mass, eps, rho, rho2):
    """Updates the dual potential by computing the optimal constant
    translation. It stabilizes and accelerates computations in sinkhorn
    loop.

    Parameters
    ----------
    u: torch.Tensor of size [Batch, size_X]
    First dual potential defined on X.

    v: torch.Tensor of size [Batch, size_Y]
    Second dual potential defined on Y.

    lcost: torch.Tensor of size [Batch, size_X, size_Y]
    Local cost depending on the current plan.

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    mass: torch.Tensor of size [Batch]
    Mass of the current plan.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    Returns
    ----------
    u: torch.Tensor of size [Batch, size_X]
    First dual potential defined on X.

    v: torch.Tensor of size [Batch, size_Y]
    Second dual potential defined on Y.
    """
    c1 = (
                 -torch.cat((u, v), 0) / (mass * rho)
                 + torch.cat((a, b), 0).log()
         ).logsumexp(dim=0) - torch.log(2 * torch.ones([1]))
    c2 = (
        (
                a.log()[:, None]
                + b.log()[None, :]
                + (
                        (u[:, None] + v[None, :] - lcost)
                        / (mass * eps)
                )
        ).logsumexp(dim=1).logsumexp(dim=0)
    )
    z = (0.5 * mass * eps) / (
            2.0 + 0.5 * (eps / rho) + 0.5 * (eps / rho2))
    k = z * (c1 - c2)
    return u + k, v + k
    
def init_plan(a, b, init=None):
    """Initialize the plan if None is given, otherwise use the input plan

    Parameters
    ----------
    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    init: torch.Tensor of size [Batch, size_X, size_Y]
    Initializes the plan. If None defaults to tensor plan.

    Returns
    ----------
    init: torch.Tensor of size [Batch, size_X, size_Y]
    Initialization of the plan to start running Sinkhorn-UGW.
    """
    if init is not None:
        return init
    else:
        return (
                a[:, None]
                * b[None, :]
                / (a.sum() * b.sum()).sqrt()
        )