import numpy as np

from hybdrt import mapping


def structure_from_x(x_drt, drtmd):
    # Get DRT and derivatives
    f = drtmd.predict_drt(psi=None, x=x_drt, tau=drtmd.tau_supergrid)
    fx = drtmd.predict_drt(psi=None, x=x_drt, tau=drtmd.tau_supergrid, order=1)
    fxx = drtmd.predict_drt(psi=None, x=x_drt, tau=drtmd.tau_supergrid, order=2)
    result = {'f': f, 'fx': fx, 'fxx': fxx}

    # Get raw probabilities
    cp = mapping.surface.peak_prob(f, fx, fxx, std_size=3, std_baseline=0.2)
    tp = mapping.surface.trough_prob(f, fx, fxx, std_size=3, std_baseline=0.2)
    p_ridge = cp * (1 - tp)
    p_trough = tp * (1 - cp)

    result['peak_prob_only'] = cp
    result['trough_prob_only'] = tp
    result['rp_raw'] = p_ridge
    result['tp_raw'] = p_trough

    # Apply ridge filters
    p_ridge_filt = mapping.surface.ridge_prob_filter(p_ridge, 0)
    p_trough_filt = mapping.surface.ridge_prob_filter(p_trough, 0)

    result['rp_filt'] = p_ridge_filt
    result['tp_filt'] = p_trough_filt

    result['rp_mix'] = 0.5 * (p_ridge + p_ridge_filt)
    result['tp_mix'] = 0.5 * (p_trough + p_trough_filt)

    return result


def structure_from_drtmd(drtmd, soc_grid, soc_dist_thresh=None, impute=True):
    groups = np.unique(drtmd.obs_group_id)

    if soc_dist_thresh is None:
        soc_dist_thresh = np.median(np.diff(soc_grid)) * 0.7

    results = {}

    for group in groups:
        index = drtmd.get_group_index(group, exclude_flagged=False)

        # Get C rate
        results[group] = {'c_rate': drtmd.obs_psi_df.loc[index, 'c_rate'].values[0]}

        # Get DRT params
        dim_grids, x_raw = mapping.ndx.assemble_ndx(
            drtmd.obs_x[index],
            drtmd.obs_psi[index], drtmd.psi_dim_names, drtmd.tau_supergrid,
            sort_by=['soc'], group_by=None,
            sort_dim_grids=[soc_grid], sort_dim_dist_thresh=[soc_dist_thresh],
            impute=False
        )
        results[group]['x_raw'] = x_raw

        dim_grids, x_drt = mapping.ndx.assemble_ndx(
            drtmd.obs_x_resolved[index],
            drtmd.obs_psi[index], drtmd.psi_dim_names, drtmd.tau_supergrid,
            sort_by=['soc'], group_by=None,
            sort_dim_grids=[soc_grid], sort_dim_dist_thresh=[soc_dist_thresh],
            impute=False
        )

        results[group]['x_res'] = x_drt

        x_filt = mapping.ndx.filter_ndx(x_drt, num_group_dims=0, by_group=False,
                                        iterative=True, iter=3, nstd=5, dev_rms_size=5,
                                        adaptive=True, impute=impute, impute_groups=False,
                                        max_sigma=(4, 1), k_factor=(4, 2),
                                        presmooth_sigma=None,
                                        mode='nearest'
                                        )
        results[group]['x_filt'] = x_filt

        rp = np.sum(x_filt, axis=-1)
        x_norm = x_filt / rp[..., None]

        # Calculate structure functions
        results[group].update(structure_from_x(x_norm, drtmd))

        # Get DOP params
        _, x_dop = mapping.ndx.assemble_ndx(
            drtmd.obs_special_resolved['x_dop'][index],
            drtmd.obs_psi[index], drtmd.psi_dim_names, drtmd.fixed_basis_nu,
            sort_by=['soc'], group_by=None,
            sort_dim_grids=[soc_grid], sort_dim_dist_thresh=[soc_dist_thresh],
            impute=False
        )

        results[group]['x_dop'] = x_dop

        x_dop_filt = mapping.ndx.filter_ndx(x_dop, num_group_dims=0, by_group=False,
                                            iterative=True, iter=3, nstd=5, dev_rms_size=5,
                                            adaptive=True, impute=impute, impute_groups=False,
                                            max_sigma=(2, 1), k_factor=(4, 2),
                                            presmooth_sigma=None,
                                            mode='nearest'
                                            )

        results[group]['x_dop_filt'] = x_dop_filt

        # Get R_inf
        _, R_inf = mapping.ndx.assemble_ndx(
            drtmd.obs_special_resolved['R_inf'][index],
            drtmd.obs_psi[index], drtmd.psi_dim_names, [0],
            sort_by=['soc'], group_by=None,
            sort_dim_grids=[soc_grid], sort_dim_dist_thresh=[soc_dist_thresh],
            impute=False
        )

        results[group]['R_inf'] = R_inf

        R_inf_filt = mapping.ndx.filter_ndx(R_inf, num_group_dims=0, by_group=False,
                                            iterative=True, iter=3, nstd=5, dev_rms_size=5,
                                            adaptive=True, impute=impute, impute_groups=False,
                                            max_sigma=(2,), k_factor=(4,),
                                            presmooth_sigma=None,
                                            mode='nearest'
                                            )

        results[group]['R_inf_filt'] = R_inf_filt

        # Shape psi to match x_drt
        _, psi = mapping.ndx.assemble_ndx(
            drtmd.obs_psi[index],
            drtmd.obs_psi[index], drtmd.psi_dim_names, drtmd.tau_supergrid,
            sort_by=['soc'], group_by=None,
            sort_dim_grids=[soc_grid], sort_dim_dist_thresh=[soc_dist_thresh],
            impute=False
        )

        # Impute nans by filtering along SOC axis
        psi = mapping.nddata.impute_nans(psi, sigma=(1, 0))

        results[group]['psi'] = psi

    return results


