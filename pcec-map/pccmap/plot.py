import numpy as np
import matplotlib.pyplot as plt

import hybdrt


def plot_x_2d(eta, x, mrt, tau=None, ax=None, cmap='coolwarm', absvmax=None, vmin=None, vmax=None, normalize=False,
              grid=True, grid_kw=None, sign_convention=-1, **kw):
    if tau is None:
        tau = mrt.tau_supergrid

    lt = np.log10(tau)
    tt, ee = np.meshgrid(lt, eta)

    if normalize:
        rp = np.nansum(np.abs(x), axis=1)
        x = x / rp[:, None]

    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 3))

    if vmin is None and vmax is None:
        if absvmax is None:
            absvmax = np.nanpercentile(np.abs(x), 99.5)
        vmin = -absvmax
        vmax = absvmax

    sm = ax.pcolormesh(tt, ee, x, cmap=cmap, vmin=vmin, vmax=vmax, **kw)

    ax.set_xlabel(r'$\log{}_{10} \tau$')
    ax.set_ylabel(r'$\eta$ (V)')

    lt_ticks = np.arange(-8, 2.1, 2)
    v_ticks = np.arange(-0.6, 0.41, 0.2)
    # Limit v_ticks to eta range
    v_ticks = v_ticks[(v_ticks >= eta[0]) & (v_ticks <= eta[-1])]

    ax.set_xticks(lt_ticks)
    ax.set_yticks(v_ticks)

    if sign_convention == -1:
        # Add small positive to prevent "-0.0" label
        ax.set_yticklabels(np.round(v_ticks * -1 + 1e-6, 1))

    if grid:
        grid_defaults = {'c': 'white', 'lw': 0.6, 'ls': ':', 'alpha': 0.5}
        if grid_kw is None:
            grid_kw = {}
        grid_kw = grid_defaults | grid_kw
        for ltt in lt_ticks:
            ax.axvline(ltt, **grid_kw)
        for vtt in v_ticks:
            ax.axhline(vtt, **grid_kw)

    return sm, ax


def plot_drt_2d(eta, x, mrt, tau=None, ax=None, cmap='coolwarm', absvmax=None, vmin=None, vmax=None, normalize=False,
                grid=True, grid_kw=None, sign_convention=-1, **kw):
    gamma = mrt.predict_drt(x=x)
    return plot_x_2d(eta, gamma, mrt, tau=tau, ax=ax, cmap=cmap, absvmax=absvmax, vmin=vmin, vmax=vmax,
                     normalize=normalize,
                     grid=grid, grid_kw=grid_kw, sign_convention=sign_convention, **kw)


def plot_dop_2d(eta, x, mrt, ax=None, normalize=True, normalize_tau=None, include_ohmic=False,
                x_ohmic=None, phase=True, grid=True, grid_kw=None, sign_convention=-1, **kw):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    nu_grid = np.linspace(-1, 1, 101)

    if normalize_tau is not None:
        quantiles = (0, 1)
    else:
        quantiles = (0.25, 0.75)

    dop = mrt.predict_dop(x=x, nu=nu_grid, normalize=normalize, normalize_tau=normalize_tau,
                          x_ohmic=x_ohmic, include_ohmic=include_ohmic, quantiles=quantiles)

    if phase:
        nu_grid = nu_grid * 90
        ax.set_xlabel(r'$-\theta \ (^\circ)$')
        x_ticks = np.arange(-90, 90.1, 45)
    else:
        ax.set_xlabel(r'$-\nu$')
        x_ticks = np.arange(-1, 1.01, 0.5)
    ax.set_xticks(x_ticks)

    v_ticks = np.arange(-0.6, 0.41, 0.2)
    # Limit v_ticks to eta range
    v_ticks = v_ticks[(v_ticks >= eta[0]) & (v_ticks <= eta[-1])]
    ax.set_yticks(v_ticks)
    ax.set_ylabel(r'$\eta$ (V)')

    if sign_convention == -1:
        # Add small positive to prevent "-0.0" label
        ax.set_yticklabels(np.round(v_ticks * -1 + 1e-6, 1))

    nn, ee = np.meshgrid(nu_grid, eta)

    sm = ax.pcolormesh(-nn, ee, dop, **kw)

    if grid:
        grid_defaults = {'c': 'white', 'lw': 0.6, 'ls': ':', 'alpha': 0.5}
        if grid_kw is None:
            grid_kw = {}
        grid_kw = grid_defaults | grid_kw
        for xt in x_ticks:
            ax.axvline(xt, **grid_kw)
        for vtt in v_ticks:
            ax.axhline(vtt, **grid_kw)

    return sm, ax


def get_group_x(ndx, group_vals, dim_grids):
    group_index = [np.where(dg == gv)[0][0] for dg, gv in zip(dim_grids, group_vals)]
    return ndx[tuple(group_index)].copy()


def plot_oh_grid(x_plot, dim_grids, mrt, tau=None, drt=False, dop=False, vabs=False,
                 absvmax=None, vmin=None, vmax=None,
                 cmap='viridis', vscale_by_group=False, axes=None, cbar=False, tight_layout=True, label_pres=True,
                 **kw):
    if axes is None:
        if cbar:
            height = 5.8
        else:
            height = 5.5
        fig, axes = plt.subplots(3, 4, figsize=(172 / 25.4, height), sharex=True, sharey=True)
    else:
        fig = axes.ravel()[0].get_figure()

    if not vabs and absvmax is not None:
        vabs = True

    v_base = dim_grids[-1]

    # if normalize:
    #     rp = np.nansum(np.abs(x_plot), axis=-1)
    #     x_plot = x_plot / rp[..., None]

    if drt:
        x_plot = mrt.predict_drt(x=x_plot, tau=tau)
        v_plot = x_plot
    elif dop:
        pred_kw = {k: v for k, v in kw.items() if k in ['normalize', 'normalize_tau', 'x_ohmic', 'include_ohmic']}
        if kw.get('normalize_tau', None) is not None:
            pred_kw['quantiles'] = (0, 1)
        else:
            pred_kw['quantiles'] = (0.25, 0.75)
        v_plot = mrt.predict_dop(x=x_plot, nu=np.linspace(-1, 1, 101), **pred_kw)
    else:
        v_plot = x_plot

    # Exclude inf values from vrange determination
    v_plot = v_plot[~np.isinf(v_plot)]

    if not vscale_by_group:
        # Set vmin and vmax (or absvmax) for entire dataset
        if vabs and absvmax is None:
            absvmax = np.nanpercentile(np.abs(v_plot), 99.5)
        else:
            if vmin is None:
                vmin = np.nanpercentile(v_plot, 0.1)
            if vmax is None:
                vmax = np.nanpercentile(v_plot, 99.9)
    # print(vmin, vmax, absvmax)

    for i, po2 in enumerate(dim_grids[0]):
        for j, ph2 in enumerate(dim_grids[1]):
            plot_group = (po2, ph2)
            x_ij = get_group_x(x_plot, plot_group, dim_grids)
            group_isnan = np.min(np.isnan(x_ij))

            if group_isnan:
                # Gray out axis
                # axes[j, i].axis('off')
                for key in axes[j, i].spines.keys():
                    axes[j, i].spines[key].set_color('grey')

                axes[j, i].tick_params(color='grey')

            # Ensure labels still applied to nan groups
            if dop:
                if 'x_ohmic' in kw.keys():
                    kw_ij = kw | {'x_ohmic': get_group_x(kw['x_ohmic'], plot_group, dim_grids)}
                else:
                    kw_ij = kw
                sm, _ = plot_dop_2d(v_base, x_ij, mrt, ax=axes[j, i], vmin=vmin, vmax=vmax,
                                    cmap=cmap, **kw_ij)
            else:
                sm, _ = plot_x_2d(v_base, x_ij, mrt, tau=tau, ax=axes[j, i], vmin=vmin, vmax=vmax,
                                  absvmax=absvmax, cmap=cmap, **kw)

            if j == 0 and label_pres:
                title = '$p_{{\mathrm{{O}}_2}} \, / \, p_{{\mathrm{{atm}}}}={:.2f}$'.format(po2)
                axes[j, i].set_title(title, size=8)

            if i == len(dim_grids[0]) - 1 and label_pres:
                title = '$p_{{\mathrm{{H}}_2}} \, / \, p_{{\mathrm{{atm}}}}={:.2f}$'.format(ph2)
                axes[j, i].text(1.05, 0.5, title,
                                transform=axes[j, i].transAxes,
                                rotation=90, va='center', size=8)

    for ax in axes[:, 1:].ravel():
        ax.set_ylabel('')

    for ax in axes[:-1].ravel():
        ax.set_xlabel('')

    if tight_layout:
        fig.tight_layout()

    if cbar:
        fig.subplots_adjust(top=0.88)
        bbox0 = axes[0, 0].get_position()
        bbox1 = axes[0, 3].get_position()
        cax = fig.add_axes([bbox0.x0, bbox0.y1 + 0.05, bbox1.x1 - bbox0.x0, 0.015])
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
    else:
        cbar = None

    return fig, axes, sm, cbar


def plot_pres_grid(x_plot, dim_grids, mrt, tau=None, drt=False, dop=False, vabs=False,
                   absvmax=None, vmin=None, vmax=None,
                   cmap='viridis', vscale_by_group=False, axes=None,
                   cbar_h=False, cbar_v=False,
                   tight_layout=True, label_pres=True,
                   **kw):
    # Plot all atmospheres in a dense grid
    if axes is None:
        if cbar_v:
            height = 3
        elif cbar_h:
            height = 4
        else:
            height = 3.5

        fig, axes = plt.subplots(2, 5, figsize=(172 / 25.4, height), sharex=True, sharey=True)
    else:
        fig = axes.ravel().get_figure()

    if not vabs and absvmax is not None:
        vabs = True

    v_base = dim_grids[-1]

    # if normalize:
    #     rp = np.nansum(np.abs(x_plot), axis=-1)
    #     x_plot = x_plot / rp[..., None]

    if drt:
        x_plot = mrt.predict_drt(x=x_plot, tau=tau)
        v_plot = x_plot
    elif dop:
        pred_kw = {k: v for k, v in kw.items() if k in ['normalize', 'normalize_tau', 'x_ohmic', 'include_ohmic']}
        if kw.get('normalize_tau', None) is not None:
            pred_kw['normalize_quantiles'] = (0, 1)
        else:
            pred_kw['normalize_quantiles'] = (0.25, 0.75)
        v_plot = mrt.predict_dop(x=x_plot, nu=np.linspace(-1, 1, 101), **pred_kw)
    else:
        v_plot = x_plot

    # Exclude inf values from vrange determination
    v_plot = v_plot[~np.isinf(v_plot)]

    if not vscale_by_group:
        # Set vmin and vmax (or absvmax) for entire dataset
        if vabs and absvmax is None:
            absvmax = np.nanpercentile(np.abs(v_plot), 99.5)
        else:
            if vmin is None:
                vmin = np.nanpercentile(v_plot, 0.1)
            if vmax is None:
                vmax = np.nanpercentile(v_plot, 99.9)
    # print(vmin, vmax, absvmax)

    ii = 0
    for i, po2 in enumerate(dim_grids[0]):
        for j, ph2 in enumerate(dim_grids[1]):
            for k, ph2o in enumerate(dim_grids[2]):
                plot_group = (po2, ph2, ph2o)
                x_ij = get_group_x(x_plot, plot_group, dim_grids)
                group_isnan = np.min(np.isnan(x_ij))

                if not group_isnan:
                    ax = axes.ravel()[ii]
                    if dop:
                        if 'x_ohmic' in kw.keys():
                            kw_ij = kw | {'x_ohmic': get_group_x(kw['x_ohmic'], plot_group, dim_grids)}
                        else:
                            kw_ij = kw
                        sm, _ = plot_dop_2d(v_base, x_ij, mrt, ax=ax, vmin=vmin, vmax=vmax,
                                            cmap=cmap, **kw_ij)
                    else:
                        sm, _ = plot_x_2d(v_base, x_ij, mrt, tau=tau, ax=ax, vmin=vmin, vmax=vmax,
                                          absvmax=absvmax, cmap=cmap, **kw)

                    if label_pres:
                        titles = []
                        for chem, px in zip(['\mathrm{O}_2', '\mathrm{H}_2', '\mathrm{H}_2 \mathrm{O}'],
                                            [po2, ph2, ph2o]):
                            # titles.append('$f_{{{}}}={:.3g}$'.format(chem, px))
                            titles.append('${:.0f}\% \, {}$'.format(100 * px, chem))
                        title = ', '.join(titles[:2]) + ',\n' + titles[2]
                        ax.set_title(title, size=7)

                    ii += 1

    for ax in axes[:, 1:].ravel():
        ax.set_ylabel('')

    for ax in axes[:-1].ravel():
        ax.set_xlabel('')

    if tight_layout:
        fig.tight_layout()

    if cbar_v:
        fig.subplots_adjust(right=0.9)
        bbox0 = axes[0, -1].get_position()
        bbox1 = axes[1, -1].get_position()
        cax = fig.add_axes([bbox0.x1 + 0.02, bbox1.y0, 0.015, bbox0.y1 - bbox1.y0])
        cbar = fig.colorbar(sm, cax=cax)

    elif cbar_h:
        if label_pres:
            top = 0.8
            vspace = 0.1
        else:
            top = 0.85
            vspace = 0.025
        fig.subplots_adjust(top=top)
        bbox0 = axes[0, 0].get_position()
        bbox1 = axes[0, -1].get_position()
        cax = fig.add_axes([bbox0.x0, bbox0.y1 + vspace, bbox1.x1 - bbox0.x0, 0.015])
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
    else:
        cbar = None

    return fig, axes, sm, cbar
