import torch
from torch.autograd import grad

from graphviz import Digraph
import numpy as np
import matplotlib.pyplot as plt

f64 = torch.float64


def main():
    plot_random_sphere_geodesic()


def plot_random_sphere_geodesic():
    x0 = torch.zeros(2, requires_grad=True)
    v0 = torch.rand(2, requires_grad=True)
    v0 = v0 / torch.norm(v0)

    dtau = 0.001
    tau_f = 2 * np.pi
    sphere_metric_geodesic_prediction(x0, v0, dtau, tau_f)


def sphere_metric_geodesic_prediction(x0, v0, dtau, tau_f):
    x = x0.clone()
    v = v0.clone()
    dvec_dtau_fn = get_dvec_dtau_fn(sphere_metric_tensor_fn)

    tau_range = torch.arange(0.0, tau_f, dtau, dtype=f64)
    X = tau_range.detach().numpy()
    position_plotter = Plotter("Position", X, r"$\tau$")
    velocity_plotter = Plotter("Velocity", X, r"$\tau$")

    thetas, phis = [], []

    for i, tau in enumerate(tau_range):
        dvdtau = dvec_dtau_fn(x.detach().requires_grad_(True), v.detach())
        v = v + dvdtau * dtau
        x = x + v * dtau

        theta = float(x[0].detach().numpy())
        phi = float(x[1].detach().numpy()) % (2 * np.pi)

        thetas.append(theta)
        phis.append(phi)
        position_plotter.record(r"$\theta$", theta)
        position_plotter.record(r"$\phi$", phi)
        velocity_plotter.record(r"$v^{\theta}$", v[0].detach().numpy())
        velocity_plotter.record(r"$v^{\phi}$", v[1].detach().numpy())

    position_plotter.plot()
    velocity_plotter.plot()
    plt.figure()

    plt.scatter(phis, thetas, s=0.1)
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\theta$")
    plt.xlim(0, 2 * np.pi)
    plt.ylim(-np.pi / 2, np.pi / 2)
    plt.axhline(y=0)
    plt.axvline(x=0)
    plt.show()


def get_dvec_dtau_fn(metric_fn, dtype=f64):
    def dvec_dtau_fn(x, vec):
        x_size = x.shape[0]
        christoffels = get_christoffels_tensor_fn(metric_fn)(x)
        dvec_dtau = torch.zeros(x_size)

        directions = range(len(x))
        for alpha in directions:
            for mu in directions:
                for nu in directions:
                    dvec_dtau[alpha] += -christoffels[alpha][mu][nu] * vec[mu] * vec[nu]
        return dvec_dtau

    return dvec_dtau_fn


def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if hasattr(var, 'variable'):
                value = str(var.variable.names)
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                name = str(type(var).__name__)
                if name != "NoneType":
                    dot.node(str(id(var)), name)
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    str1 = str(id(u[0]))
                    str2 = str(id(var))
                    if None not in [var, u[0]]:
                        dot.edge(str1, str2)
                    add_nodes(u[0])

    add_nodes(var.grad_fn)
    return dot


def plot_sphere_tensors():
    plot_tensors(sphere_metric_tensor_fn)


def plot_minkowski_tensors():
    plot_tensors(minkowski_metric_fn)


def plot_tensors(metric_fn, dtype=f64):
    metric_fn = metric_fn
    christoffels_fn = get_christoffels_tensor_fn(metric_fn)
    d_christoffels_fn = grad_fn(christoffels_fn)
    rct_fn = get_riemann_curvature_tensor_fn(metric_fn)
    theta_range = torch.arange(-np.pi, np.pi, 0.05, dtype=dtype)
    X = theta_range.detach().numpy()

    d_theta = r"\frac{d}{d\theta}"
    christoffel_plotter = Plotter("Christoffel Symbols", X, r"\theta")
    dchristoffel_plotter = Plotter(r"$" + d_theta + r"Christoffel Symbols$", X, r"\theta")
    rct_plotter = Plotter("Riemann Curvature Tensor", X, r"$\theta$")

    for theta in theta_range:
        xi = torch.tensor([theta, 0.0], requires_grad=True, dtype=dtype)

        christoffels = christoffels_fn(xi)
        christoffel_plotter.record(r"${\Gamma^{\theta}}_{\phi\phi}$", christoffels[0, 1, 1].detach().numpy())
        christoffel_plotter.record(r"${\Gamma^{\phi}}_{\theta\phi}$", christoffels[1, 0, 1].detach().numpy())

        dchristoffels = d_christoffels_fn(xi)
        dchristoffel_plotter.record(r"$" + d_theta + r"{\Gamma^{\theta}}_{\phi\phi}$",
                                    dchristoffels[0, 1, 1, 0].detach().numpy())
        dchristoffel_plotter.record(r"$" + d_theta + r"{\Gamma^{\phi}}_{\theta\phi}$",
                                    dchristoffels[1, 0, 1, 0].detach().numpy())

        rct = rct_fn(xi)
        rct_plotter.record(r"${R^{\theta}}_{\phi\theta\phi}$", rct[0, 1, 0, 1].detach().numpy())
        rct_plotter.record(r"${R^{\theta}}_{\phi\phi\theta}$", rct[0, 1, 1, 0].detach().numpy())
        rct_plotter.record(r"${R^{\phi}}_{\theta\theta\phi}$", rct[1, 0, 0, 1].detach().numpy())
        rct_plotter.record(r"${R^{\phi}}_{\theta\phi\theta}$", rct[1, 0, 1, 0].detach().numpy())

    christoffel_plotter.plot()
    dchristoffel_plotter.plot()
    rct_plotter.plot()


def get_riemann_curvature_tensor_fn(metric_fn):
    def riemann_curvature_fn(x):
        x_size = x.shape[0]

        christoffels_fn = get_christoffels_tensor_fn(metric_fn)
        d_christoffels_fn = grad_fn(christoffels_fn)

        christoffels = christoffels_fn(x)
        d_christoffels = d_christoffels_fn(x)

        rct_shape = (x_size,) + tuple(christoffels.shape)
        rct = torch.zeros(rct_shape, requires_grad=True)

        directions = range(len(x))

        for alpha in directions:
            for beta in directions:
                for mu in directions:
                    for nu in directions:
                        t1 = d_christoffels[alpha][beta][nu][mu]
                        t2 = torch.zeros_like(t1)
                        t3 = d_christoffels[alpha][beta][mu][nu]
                        t4 = torch.zeros_like(t3)
                        # Sum over gamma
                        for gamma in directions:
                            t2 += christoffels[gamma][beta][nu] * christoffels[alpha][gamma][mu]
                            t4 += christoffels[gamma][beta][mu] * christoffels[alpha][gamma][nu]
                        rct[alpha][beta][mu][nu] = t1 + t2 - t3 - t4
        return rct

    return riemann_curvature_fn


def get_ricci_tensor_fn(metric_fn):
    def ricci_tensor_fn(x):
        x_size = x.shape[0]
        rct_fn = get_riemann_curvature_tensor_fn(metric_fn)
        rct = rct_fn(x)
        ricci_tensor = torch.zeros((x_size, x_size))

        directions = range(len(x))
        for mu in directions:
            for nu in directions:
                acc = torch.zeros(1)
                for gamma in directions:
                    acc += rct[gamma][mu][gamma][nu]
                ricci_tensor[mu][nu] = acc
        return ricci_tensor

    return ricci_tensor_fn


def get_ricci_scalar_fn(metric_fn):
    def ricci_scalar_fn(x):
        g = metric_fn(x)
        ricci_tensor = get_ricci_tensor_fn(metric_fn)(x)
        ricci_scalar = torch.zeros(1)

        directions = range(len(x))
        for mu in directions:
            ricci_scalar += ricci_tensor[mu][mu] / g[mu][mu]

        return ricci_scalar

    return ricci_scalar_fn


def get_christoffels_tensor_fn(metric_fn, dtype=f64):
    def christoffels_tensor_fn(x):
        g = metric_fn(x)
        dg = grad_fn(metric_fn)(x)

        christoffels = torch.zeros_like(dg, dtype=dtype)
        directions = range(len(x))

        for gamma in directions:
            for alpha in directions:
                for beta in directions:
                    # TODO: find better name than triplet
                    triplet = dg[gamma][alpha][beta] + dg[gamma][beta][alpha] - dg[alpha][beta][gamma]
                    christoffels[gamma][alpha][beta] = triplet / (2.0 * g[gamma][gamma])
        return christoffels

    return christoffels_tensor_fn


def sphere_metric_tensor_fn(x):
    r = torch.ones(1, dtype=f64)
    r2 = r * r
    theta, phi = x
    r2c = r2 * torch.cos(theta) * torch.cos(theta)
    return torch.diag(torch.hstack([r2, r2c]))


def minkowski_metric_fn(x):
    r = torch.ones(1, dtype=f64)
    r2 = r * r
    return torch.diag(torch.hstack([r2, -torch.ones(1, requires_grad=True)]))


def grad_fn(f):
    def df(x):
        x_size = x.shape[0]
        x_shape = x.shape
        x_ = x.clone().requires_grad_(True)
        result = f(x_)
        grad_size = (x_size,) + tuple(result.shape)
        out = torch.zeros(grad_size)
        result_vec = result.reshape(-1, 1)
        out_vec = out.reshape(-1, x_size)
        for idx in range(len(result_vec)):
            grad_value = grad(
                outputs=result_vec[idx],
                inputs=x,
                grad_outputs=torch.ones(1),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad_value is None:
                grad_value = torch.zeros(x_shape)
            out_vec[idx] = grad_value
        out = out_vec.reshape(grad_size)
        return out

    return df


class Plotter:
    def __init__(self, title, X, X_name="", Y_name=""):
        self.title = title
        self.X = X
        self.X_name = X_name
        self.Y_name = Y_name
        self.values_to_plot = {}

    def record(self, name, value):
        if name in self.values_to_plot:
            self.values_to_plot[name].append(value)
        else:
            self.values_to_plot[name] = [value]

    def plot(self):
        plt.figure()
        for name in self.values_to_plot:
            plt.plot(self.X, self.values_to_plot[name], label=name)
        plt.legend()
        # plt.axvline(x=0, color='k')
        # plt.axhline(y=0, color='k')
        plt.xlabel(self.X_name)
        plt.ylabel(self.Y_name)
        plt.title(self.title)
        plt.show()


if __name__ == '__main__':
    main()
