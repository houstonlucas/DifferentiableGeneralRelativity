import time

from DGR.DGR import get_dvec_dtau_fn
from DGR.DGR import Plotter
from DGR.DGR import get_ricci_scalar_fn
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f64 = torch.float64

G = torch.tensor(6.6743e-11)
c = torch.tensor(2.99792e8)

minute = 60.0
hour = 60 * minute
day = 24 * hour
year = 365 * day


def main():
    integrate_geodesic_forward(dtau=5 * minute, tau_final=0.5 * year)
    # heatmap_of_ricci_scalar()


def heatmap_of_ricci_scalar():
    fig = plt.figure()
    ax = Axes3D(fig)

    n = 10
    m = 30
    rs = np.linspace(3e4, 2e7, m)
    thetas = np.linspace(0, 2 * np.pi, n)
    r_grid, theta_grid = np.meshgrid(rs, thetas)

    z = np.zeros((n, m))

    ricci_scalar_fn = get_ricci_scalar_fn(schwarzschild_metric_fn)

    for r_idx in range(rs.shape[0]):
        for theta_idx in range(thetas.shape[0]):
            x = torch.tensor([0.0, rs[r_idx], thetas[theta_idx]], requires_grad=True)
            z[theta_idx][r_idx] = ricci_scalar_fn(x).detach().numpy() * 1e18

    plt.subplot(projection="polar")
    plt.pcolormesh(theta_grid, r_grid, z)
    plt.plot(thetas, r_grid, ls='none', color='k')
    plt.grid()
    plt.colorbar()
    plt.show()


def integrate_geodesic_forward(dtau, tau_final, num_iters_between_plot=100):
    x0 = torch.zeros(3, requires_grad=False)
    v0 = torch.zeros(3, requires_grad=False)

    # # Earth Aphelion (m)
    # r_aphelion = 1.521e11
    # x0[1] = r_aphelion
    # v0[0] = c
    # v_aphelion = 2.929e4
    # v0[2] = v_aphelion / r_aphelion

    # Mercury Aphelion (m)
    r_aphelion = 6.9817e10
    x0[1] = r_aphelion
    v0[0] = c
    v_aphelion = 3.886e4
    v0[2] = v_aphelion / r_aphelion

    print(
        "\n".join([
            f"Mercury Orbit GR Simulation\n",
            f"Aphelion start: {r_aphelion / 1e10:.3f}1e10 m"
        ])
    )

    dvec_dtau_fn = get_dvec_dtau_fn(schwarzschild_metric_fn)

    tau_range = torch.arange(0.0, tau_final, dtau, dtype=f64)
    X = tau_range.detach().numpy()
    position_plotter = Plotter("Position", X[::num_iters_between_plot], X_name=r"$\tau$", Y_name=r"$r$, $\theta$")
    velocity_plotter = Plotter("Velocity", X[::num_iters_between_plot], X_name=r"$\tau$", Y_name=r"$v^{r}$, $v^{\theta}$")

    ts, rs, thetas = [], [], []
    drdtaus, dthetadtaus = [], []

    x = x0.clone()
    v = v0.clone()

    # TODO: Use https://github.com/rtqichen/torchdiffeq for integrating forward
    start = time.time()
    for i, tau in enumerate(tau_range):
        dvdtau = dvec_dtau_fn(x.detach().requires_grad_(True), v.detach())
        v = (v + dvdtau * dtau).detach()
        x = (x + v * dtau).detach()

        t = float(x[0].detach().numpy())
        r = float(x[1].detach().numpy()) / 1e10  # r is scaled to plot nicely with theta
        theta = float(x[2].detach().numpy()) % (2 * np.pi)

        if i % num_iters_between_plot == 0:
            # print("\r{tau / tau_final}")
            drdtaus.append(v[1]/1e4)
            dthetadtaus.append(v[2]*1e6)
            ts.append(t)
            rs.append(r)
            thetas.append(theta)

            # position_plotter.record(r"$t$", t)
            position_plotter.record(r"$r$ 1e10 m", r)
            position_plotter.record(r"$\theta$ Rad", theta)

            # velocity_plotter.record(r"$v^{t}$", v[0].detach().numpy())
            velocity_plotter.record(r"$v^{r}$ 1e4 m", v[1].detach().numpy() / 1e4)
            velocity_plotter.record(r"$v^{\theta}$ 1e-6 Rad/s", v[2].detach().numpy() * 1e6)

    dur = time.time() - start
    print(f"Simulation Took {dur:.2f} secconds.")
    print("\n".join([
        f"Simulation results: "
        f"- Perihelion: {min(rs):.3f}e10 m",
        f"- Max radial velocity reached: {max(drdtaus):.3f} 1e4 m/s",
        f"- Min radial velocity reached: {min(drdtaus):.3f} 1e4 m/s",
        f"- Max angular velocity reached: {max(dthetadtaus):.3f} 1e-6 rad/s",
        f"- Min angular velocity reached: {min(dthetadtaus):.3f} 1e-6 rad/s"
    ]))

    position_plotter.plot()
    velocity_plotter.plot()

    plt.figure()
    plt.polar(thetas, rs)
    plt.xlabel("Radial units 1e10 m")
    plt.show()


def schwarzschild_metric_fn(x):
    t, r, theta = x
    r2 = r * r
    solar_mass = torch.tensor(1.988e30)
    c2 = c * c
    r_s = 2 * G * solar_mass / c2
    z = (1 - (r_s / r))
    return torch.diag(torch.hstack([-z, 1 / z, r2]))


if __name__ == '__main__':
    main()
