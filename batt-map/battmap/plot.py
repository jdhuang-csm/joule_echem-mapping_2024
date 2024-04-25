import numpy as np
import matplotlib.pyplot as plt


def plot_x_2d(x, soc_grid, mrt, tau=None, ax=None, **kw):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    if tau is None:
        tau = mrt.tau_supergrid

    lt = np.log10(tau)
    tt, ss = np.meshgrid(lt, soc_grid)

    sm = ax.pcolormesh(tt, ss, x, **kw)

    # Tick formatting
    ax.set_xticks(np.arange(-7, 1.1, 2))

    soc_ticks = np.linspace(0.5, 1, 6)
    soc_ticklabels = (100 * soc_ticks).astype(int)
    ax.set_yticks(soc_ticks)
    ax.set_yticklabels(soc_ticklabels)

    ax.set_xlim(-7, 2)
    ax.set_ylim(0.5, 1.)

    # Labels
    ax.set_xlabel(r'$\log_{10}{\tau}$')
    ax.set_ylabel('SOC (%)')

    return sm, ax


def plot_drt_2d(x, soc_grid, mrt, tau=None, ax=None, order=0, **kw):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    if tau is None:
        tau = mrt.tau_supergrid
    lt = np.log10(tau)
    tt, ss = np.meshgrid(lt, soc_grid)

    gamma = mrt.predict_drt(None, x=x, tau=tau, order=order)

    sm = ax.pcolormesh(tt, ss, gamma, **kw)

    # Tick formatting
    ax.set_xticks(np.arange(-7, 1.1, 2))

    soc_ticks = np.linspace(0.5, 1, 6)
    soc_ticklabels = (100 * soc_ticks).astype(int)
    ax.set_yticks(soc_ticks)
    ax.set_yticklabels(soc_ticklabels)

    ax.set_xlim(-7, 2)
    ax.set_ylim(0.5, 1.)

    # Labels
    ax.set_xlabel(r'$\log_{10}{\tau}$')
    ax.set_ylabel('SOC (%)')

    return sm, ax


def plot_dop_2d(x, soc_grid, mrt, ax=None, normalize=True, normalize_tau=None, include_ohmic=False,
                x_ohmic=None, nu_grid=None, phase=True, **kw):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    if nu_grid is None:
        nu_grid = np.linspace(-1, 1, 201)

    if normalize_tau is not None:
        quantiles = (0, 1)
    else:
        quantiles = (0.25, 0.75)
    dop = mrt.predict_dop(x=x, nu=nu_grid, normalize=normalize, normalize_tau=normalize_tau,
                          x_ohmic=x_ohmic, include_ohmic=include_ohmic, normalize_quantiles=quantiles)

    if phase:
        nu_grid = nu_grid * 90

    nn, ss = np.meshgrid(nu_grid, soc_grid)
    sm = ax.pcolormesh(-nn, ss, dop, **kw)

    # Tick formatting
    ax.set_xticks(np.arange(-90, 90.1, 45))

    soc_ticks = np.linspace(0.5, 1, 6)
    soc_ticklabels = (100 * soc_ticks).astype(int)
    ax.set_yticks(soc_ticks)
    ax.set_yticklabels(soc_ticklabels)

    # Labels
    if phase:
        ax.set_xlabel(r'$-\theta \ (^\circ)$')
    else:
        ax.set_xlabel(r'$-\nu$')
    ax.set_ylabel('SOC (%)')

    return sm, ax
