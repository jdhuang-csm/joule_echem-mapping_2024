import numpy as np
from copy import deepcopy
from scipy import ndimage

from hybdrt import mapping
from hybdrt.mapping import flow, ndx
from hybdrt.mapping.nddata import impute_nans, flag_bad_obs
from hybdrt.filters import rog_filter


def flag_bad_by_group(drtmd, groups=None, median_size=(5, 3), std_size=(5, 3), thresh=2):
    if groups is None:
        groups = np.unique(drtmd.obs_group_id)

    for group in groups:
        index = drtmd.get_group_index(group)

        x_raw = drtmd.obs_x[index]
        eta_vals = drtmd.obs_psi[index, drtmd.psi_dim_names.index('eta')]
        sort_index = np.argsort(eta_vals)
        # eta_vals = eta_vals[sort_index]
        x_raw = x_raw[sort_index]
        index = index[sort_index]
        x_filt = ndimage.median_filter(x_raw, size=median_size)

        bad_flag = flag_bad_obs(x_raw, x_filt, std_size=std_size, thresh=thresh,
                                test_factor_correction=False, test_offset_correction=False,
                                return_rss=False, robust_std=True)

        bad_obs_flag = np.max(bad_flag, axis=1)
        drtmd.obs_ignore_flag[index] = bad_obs_flag
        # print(np.where(bad_obs_flag), eta_vals[bad_obs_flag])

        print(f'{group} num bad:', np.sum(np.max(bad_flag, axis=1)))


def structure_from_x(x_drt, drtmd, num_group_dims, drt_var=None):
    # Get DRT and derivatives
    f = drtmd.predict_drt(psi=None, x=x_drt, tau=drtmd.tau_supergrid)
    fx = drtmd.predict_drt(psi=None, x=x_drt, tau=drtmd.tau_supergrid, order=1)
    fxx = drtmd.predict_drt(psi=None, x=x_drt, tau=drtmd.tau_supergrid, order=2)
    result = {'f': f, 'fx': fx, 'fxx': fxx}

    # Get raw probabilities
    prob_kw = dict(std_size=3, std_baseline=0.2)
    if drt_var is not None:
        drt_std = {k: v ** 0.5 for k, v in drt_var.items()}
        prob_kw.update(f_var=(drt_std[0] + 0.0 * np.nanstd(drt_std[0])) ** 2,
                       fx_var=None,  # (drt_std[1] + 0.0 * np.nanstd(drt_std[1])) ** 2,
                       fxx_var=(drt_std[2] + 0.0 * np.nanstd(drt_std[2])) ** 2,
                       )
    cp = mapping.surface.peak_prob(f, fx, fxx, **prob_kw)
    tp = mapping.surface.trough_prob(f, fx, fxx, **prob_kw)
    p_ridge = cp * (1 - tp)
    p_trough = tp * (1 - cp)

    result['peak_prob_only'] = cp
    result['trough_prob_only'] = tp
    result['rp_raw'] = p_ridge
    result['tp_raw'] = p_trough

    # Apply ridge filters
    p_ridge_filt = mapping.surface.ridge_prob_filter(p_ridge, num_group_dims)
    p_trough_filt = mapping.surface.ridge_prob_filter(p_trough, num_group_dims)

    result['rp_filt'] = p_ridge_filt
    result['tp_filt'] = p_trough_filt

    result['rp_mix'] = 0.5 * (p_ridge + p_ridge_filt)
    result['tp_mix'] = 0.5 * (p_trough + p_trough_filt)

    return result


def structure_from_drtmd(drtmd, eta_grid, eta_dist_thresh=None, impute=True, group_by=None, sigma=None,
                         remove_bad=False, flag_kw=None, resolve=False, resolve_kw=None):
    if group_by is None:
        group_by = ['po2', 'ph2', 'ph2o']
    if sigma is None:
        sigma = (2, 1)

    temps = np.unique(drtmd.obs_psi_df['T'])

    if eta_dist_thresh is None:
        eta_dist_thresh = np.median(np.diff(eta_grid)) * 0.7

    if remove_bad:
        if flag_kw is None:
            flag_kw = {}
        flag_bad_by_group(drtmd, **flag_kw)
    if resolve:
        for group in np.unique(drtmd.obs_group_id):
            if resolve_kw is None:
                resolve_kw = dict(psi_sort_dims=['eta'], sigma=0.75, lambda_psi=75,
                                  tau_filter_sigma=0., special_filter_sigma=0.)
            # print('resolve_kw:', resolve_kw)
            drtmd.resolve_group(group, **resolve_kw)

    results = {}

    for temp in temps:
        index = drtmd.filter_psi({'T': temp}, exclude_flagged=True)
        results[temp] = {}

        # Get DRT params
        dim_grids, x_raw = mapping.ndx.assemble_ndx(
            drtmd.obs_x[index],
            drtmd.obs_psi[index], drtmd.psi_dim_names, drtmd.tau_supergrid,
            sort_by=['eta'], group_by=group_by,
            sort_dim_grids=[eta_grid], sort_dim_dist_thresh=[eta_dist_thresh],
            impute=False
        )
        results[temp]['x_raw'] = x_raw

        dim_grids, x_drt = mapping.ndx.assemble_ndx(
            drtmd.obs_x_resolved[index],
            drtmd.obs_psi[index], drtmd.psi_dim_names, drtmd.tau_supergrid,
            sort_by=['eta'], group_by=group_by,
            sort_dim_grids=[eta_grid], sort_dim_dist_thresh=[eta_dist_thresh],
            impute=False
        )

        results[temp]['x_res'] = x_drt
        results[temp]['dim_grids'] = dim_grids

        x_filt = mapping.ndx.filter_ndx(x_drt, num_group_dims=len(group_by), by_group=True,
                                        iterative=True, iter=3, nstd=5, dev_rms_size=5,
                                        adaptive=True, impute=impute, impute_groups=False,
                                        max_sigma=sigma, k_factor=(4, 2),
                                        presmooth_sigma=None,
                                        mode='nearest'
                                        )
        results[temp]['x_filt'] = x_filt

        Rp = np.sum(np.abs(x_filt), axis=-1)
        x_norm = x_filt / Rp[..., None]
        results[temp]['x_norm'] = x_norm

        #
        _, obs_index = mapping.ndx.assemble_ndx(
            index.astype(float),
            drtmd.obs_psi[index], drtmd.psi_dim_names, [0],
            sort_by=['eta'], group_by=group_by,
            sort_dim_grids=[eta_grid], sort_dim_dist_thresh=[eta_dist_thresh],
            impute=False
        )

        # def predict_var(obs_id, var_order):
        #     if np.isnan(obs_id):
        #         out = np.empty(len(drtmd.tau_supergrid))
        #         out.fill(np.nan)
        #     else:
        #         return drtmd.predict_drt_var(int(obs_id), tau=drtmd.tau_supergrid, order=var_order)
        #
        # drt_var = {}
        # norm_var = {}
        # for order in [0, 1, 2]:
        #     var = np.empty_like(x_filt)
        #     var.fill(np.nan)
        #     it = np.nditer(var, op_axes=[list(np.arange(len(group_by) + 1))], flags=['multi_index'])
        #     for _ in it:
        #         ijk = it.multi_index
        #         var[ijk] = predict_var(obs_index[ijk], var_order=order)
        #
        #     var = mapping.ndx.filter_ndx(var, num_group_dims=len(group_by), by_group=True,
        #                                  iterative=True, iter=3, nstd=5, dev_rms_size=5,
        #                                  adaptive=True, impute=impute, impute_groups=False,
        #                                  max_sigma=(1, 0), k_factor=(4, 2),
        #                                  presmooth_sigma=None,
        #                                  mode='nearest'
        #                                  )
        #     drt_var[order] = var
        #     norm_var[order] = var / Rp[..., None] ** 2
        #
        # results[temp]['drt_var'] = drt_var
        # results[temp]['norm_var'] = norm_var

        # _, drt_var = drtmd.predict_drt_var(index)

        # Calculate structure functions
        results[temp].update(structure_from_x(x_norm, drtmd, len(group_by), drt_var=None))

        # Get DOP params
        _, x_dop = mapping.ndx.assemble_ndx(
            drtmd.obs_special_resolved['x_dop'][index],
            drtmd.obs_psi[index], drtmd.psi_dim_names, drtmd.fixed_basis_nu,
            sort_by=['eta'], group_by=group_by,
            sort_dim_grids=[eta_grid], sort_dim_dist_thresh=[eta_dist_thresh],
            impute=False
        )

        results[temp]['x_dop'] = x_dop

        x_dop_filt = mapping.ndx.filter_ndx(x_dop, num_group_dims=len(group_by), by_group=True,
                                            iterative=True, iter=3, nstd=5, dev_rms_size=5,
                                            adaptive=True, impute=impute, impute_groups=False,
                                            max_sigma=sigma, k_factor=(4, 2),
                                            presmooth_sigma=None,
                                            mode='nearest'
                                            )

        results[temp]['x_dop_filt'] = x_dop_filt

        # Get R_inf
        _, R_inf = mapping.ndx.assemble_ndx(
            drtmd.obs_special_resolved['R_inf'][index],
            drtmd.obs_psi[index], drtmd.psi_dim_names, [0],
            sort_by=['eta'], group_by=group_by,
            sort_dim_grids=[eta_grid], sort_dim_dist_thresh=[eta_dist_thresh],
            impute=False
        )

        R_inf = np.squeeze(R_inf, axis=-1)
        results[temp]['R_inf'] = R_inf

        R_inf_filt = mapping.ndx.filter_ndx(R_inf, num_group_dims=len(group_by), by_group=True,
                                            iterative=True, iter=3, nstd=5, dev_rms_size=5,
                                            adaptive=True, impute=impute, impute_groups=False,
                                            max_sigma=(sigma[0],), k_factor=(4,),
                                            presmooth_sigma=None,
                                            mode='nearest'
                                            )

        results[temp]['R_inf_filt'] = R_inf_filt

        # Shape psi to match x_drt
        _, psi = mapping.ndx.assemble_ndx(
            drtmd.obs_psi[index],
            drtmd.obs_psi[index], drtmd.psi_dim_names, np.ones(len(drtmd.psi_dim_names)),
            sort_by=['eta'], group_by=group_by,
            sort_dim_grids=[eta_grid], sort_dim_dist_thresh=[eta_dist_thresh],
            impute=False
        )

        # Impute psi nans by filtering along eta axis
        impute_sigma = (0,) * len(group_by) + (1, 0)
        psi = mapping.nddata.impute_nans(psi, sigma=impute_sigma)

        results[temp]['psi'] = psi

    return results


# Flow/warping
def partial_flow_rog(ref, moving, sigma_loc, sigma_glob, **flow_kw):
    img_stack = np.stack((ref, moving), axis=0)
    img_stack = rog_filter(img_stack, sigma_loc, sigma_glob)
    ref = img_stack[0]
    moving = img_stack[1]

    return flow.partial_flow_ilk(ref, moving, **flow_kw)


def index_nested(nested_list, index):
    out = nested_list
    for i in index:
        # print(i, len(out))
        out = out[i]
    return out


# def align_conditions_multitemp(x, num_group_dims, condition_ref_index,
#                                gaussian=True, prefilter=True, bidirectional=False):


def align_single_temp(x, group_by, ref_index, gaussian=True, prefilter=True, bidirectional=False,
                      rog=False, rog_kw=None,
                      group_radius=(9, 9), group_sigma=None,
                      momentum_radius=1, momentum_sigma=0.5,
                      coop_radius=1, coop_sigma=0.5,
                      coop_h2_warp=False,
                      **flow_kw):
    x_align = x.copy()
    num_group_dims = len(group_by)

    if group_sigma is None:
        group_sigma = tuple(np.array(group_radius) / 2)

    if bidirectional:
        solver = flow.bidirectional_flow
    else:
        solver = flow.partial_flow_ilk

    if rog:
        mask = ~np.isnan(x)
        if rog_kw is None:
            rog_kw = dict(sigma_loc=(0,) * num_group_dims + (1, 5),
                          sigma_glob=(1,) * num_group_dims + (0, 0),
                          mask=mask)
        x_input = rog_filter(x, **rog_kw)
    else:
        x_input = x.copy()

    # x_input = x_input#.copy()

    # For now, flow is assumed to be only along one axis
    # Then the flow array from one group to the next has the same shape as the group array
    # flow_shape = np.array(x_align.shape)
    # flow_shape[:num_group_dims] -= 1
    # flows = np.zeros(tuple(flow_shape))
    flows = np.zeros(x.shape)

    # Construct flow sequence container with same shape as x group dims (one list for each group)
    flow_sequence = None
    for n in range(-(np.ndim(x) - num_group_dims) - 1, -np.ndim(x) - 1, -1):
        if flow_sequence is None:
            flow_sequence = [[] for i in range(x.shape[n])]
        else:
            flow_sequence = [deepcopy(flow_sequence) for i in range(x.shape[n])]

    # tmp = flow_sequence
    # for n in range(np.ndim(x)):
    #     print(n, len(tmp))
    #     tmp = tmp[0]

    group_exists = ~ndx.group_isnan(x, num_group_dims)

    flow_axes = (-1,)

    ph2o_axis = group_by.index('ph2o')
    ph2_axis = group_by.index('ph2')
    po2_axis = group_by.index('po2')

    # First compress ph2o
    # ph2o_vals = dim_grids[ph2o_axis]
    for group_index in np.argwhere(group_exists):
        if group_index[ph2o_axis] != ref_index[ph2o_axis]:
            print(group_index)
            x_group_input = x_input[tuple(group_index)]
            x_group = x_align[tuple(group_index)]

            target_index = group_index.copy()
            target_index[ph2o_axis] = ref_index[ph2o_axis]

            x_target = x_input[tuple(target_index)]

            flow_ = solver(x_target, x_group_input, flow_axes=flow_axes,
                           gaussian=gaussian, prefilter=prefilter,
                           radius=group_radius, sigma=group_sigma,
                           **flow_kw)
            flows[tuple(group_index)] = flow_[flow_axes]
            index_nested(flow_sequence, group_index).append(flow_)
            x_align[tuple(group_index)] = flow.warp(x_group, flow_)
            x_input[tuple(group_index)] = flow.warp(x_group_input, flow_)

    # Next collapse low and high pH2 along po2 axis
    # TODO: set this up for variable group_by. Currently hard-coded for [po2, ph2, ph2o]
    h2_flow_weights = np.zeros((2, x.shape[po2_axis]))
    h2_flows = np.zeros((2, x.shape[po2_axis], *x.shape[num_group_dims:]))
    h2_flows_full = np.empty((x.shape[ph2_axis], x.shape[po2_axis], np.ndim(x) - num_group_dims,
                              *x.shape[num_group_dims:]))
    h2_flows_full.fill(np.nan)
    for j in [0, -1]:
        x_in_h2 = []
        x_h2 = []
        h2_index = []
        counter = 0
        for i in range(x.shape[po2_axis]):
            ii = (i, j, ref_index[ph2o_axis])
            print(i, j, ii)
            if group_exists[ii]:
                print(x[ii].shape)
                x_in_h2.append(x_input[ii])
                x_h2.append(x_align[ii])
                h2_index.append(i)
                h2_flow_weights[j, i] = 1
                # Count up to the po2 reference index, since some groups may not exist
                if i == ref_index[po2_axis]:
                    po2_ref = deepcopy(counter)

                counter += 1

        print('po2_ref:', po2_ref)

        x_stack = np.stack(x_h2, axis=0)
        x_input_stack = np.stack(x_in_h2, axis=0)
        flow_h2 = solver(x_input_stack[1:], x_input_stack[:-1], flow_axes=flow_axes,
                         gaussian=gaussian, prefilter=prefilter,
                         radius=(momentum_radius,) + group_radius, sigma=(momentum_sigma,) + group_sigma,
                         **flow_kw)

        x_, flow_ = flow.align_to_ref(x_stack, flow_h2, po2_ref, axis=0, return_flows=True)
        x_input_ = flow.align_to_ref(x_input_stack, flow_h2, po2_ref, axis=0)

        for stack_i, i in enumerate(h2_index):
            print(i, stack_i)
            ijk = (i, j, ref_index[ph2o_axis])
            x_align[ijk] = x_[stack_i]
            x_input[ijk] = x_input_[stack_i]
            if flow_[stack_i] is not None:
                flows[ijk] += flow_[stack_i][-1]
                index_nested(flow_sequence, ijk).append(flow_[stack_i])

                h2_flows[j, i] = flow_[stack_i][-1]

                # h2_flows_full[j, i, :3] = 0  # Missing dims since x_stack is a stack of 2d frames
                h2_flows_full[j, i] = flow_[stack_i]

                # Apply po2 flow to other ph2o layers
                for k in range(x.shape[ph2o_axis]):
                    if k != ref_index[ph2o_axis]:
                        flows[i, j, k] += flow_[stack_i][-1]
                        index_nested(flow_sequence, (i, j, k)).append(flow_[stack_i])
                        if group_exists[i, j, k]:
                            x_align[i, j, k] = flow.warp(x_align[i, j, k], flow_[stack_i])
                            x_input[i, j, k] = flow.warp(x_input[i, j, k], flow_[stack_i])
            else:
                h2_flows[j, i] = 0
                h2_flows_full[j, i] = 0

    # Apply po2 flow to central ph2 value
    # top_flow = flows[:, 0, ref_index[ph2o_axis]]
    # bot_flow = flows[:, -1, ref_index[ph2o_axis]]
    top_flow = h2_flows[0]
    bot_flow = h2_flows[1]
    # TODO: impute mean_h2_flow
    h2_flow_weights = h2_flow_weights / np.sum(h2_flow_weights, axis=0)[None, :]
    mean_h2_flow = top_flow * h2_flow_weights[0][:, None, None] + bot_flow * h2_flow_weights[1][:, None, None]
    # Impute flows for central ph2 value, filtering across ph2 only
    h2_flows_full = impute_nans(h2_flows_full, sigma=(0.5,) + (0,) * (np.ndim(h2_flows_full) - 1))

    flows[:, 1, ref_index[ph2o_axis]] += mean_h2_flow
    for i in range(x.shape[po2_axis]):
        index_nested(flow_sequence, (i, 1, ref_index[ph2o_axis])).append(h2_flows_full[1, i])

    for i in range(x.shape[po2_axis]):
        ijk = (i, 1, ref_index[ph2o_axis])
        # Construct full flow field for all dims
        if group_exists[ijk]:
            full_flow = np.zeros((2, *x_align[ijk].shape))
            full_flow[-1] = mean_h2_flow[i]
            x_align[ijk] = flow.warp(x_align[ijk], full_flow)
            x_input[ijk] = flow.warp(x_input[ijk], full_flow)

    # Apply po2 flow for central ph2 value to other ph2o layers
    for k in range(x.shape[ph2o_axis]):
        if k != ref_index[ph2o_axis]:
            flows[:, 1, k] += mean_h2_flow
            for i in range(x.shape[po2_axis]):
                ijk = (i, 1, k)
                index_nested(flow_sequence, ijk).append(h2_flows_full[1, i])
                if group_exists[ijk]:
                    full_flow = np.zeros((2, *x_align[ijk].shape))
                    full_flow[-1] = mean_h2_flow[i]
                    x_align[ijk] = flow.warp(x_align[ijk], full_flow)
                    x_input[ijk] = flow.warp(x_input[ijk], full_flow)

    # Finally, collapse along ph2 axis
    if coop_h2_warp:
        x_lo = []
        x_input_lo = []
        x_mid = []
        x_input_mid = []
        x_hi = []
        x_input_hi = []
        o2_index = []
        for i in range(x.shape[po2_axis]):
            if group_exists[i, 0, ref_index[ph2o_axis]] and group_exists[i, -1, ref_index[ph2o_axis]]:
                x_lo.append(x_align[i, 0, ref_index[ph2o_axis]])
                x_input_lo.append(x_input[i, 0, ref_index[ph2o_axis]])
                x_hi.append(x_align[i, -1, ref_index[ph2o_axis]])
                x_input_hi.append(x_input[i, -1, ref_index[ph2o_axis]])

                if group_exists[i, 1, ref_index[ph2o_axis]]:
                    # If group exists at middle ph2 for this po2, use that group
                    x_mid.append(x_align[i, 1, ref_index[ph2o_axis]])
                    x_input_mid.append(x_input[i, 1, ref_index[ph2o_axis]])
                else:
                    # Otherwise use the standard po2 group at middle ph2
                    # (since all groups have already been warped to the same effective po2)
                    x_mid.append(x_align[2, 1, ref_index[ph2o_axis]])
                    x_input_mid.append(x_input[2, 1, ref_index[ph2o_axis]])

                o2_index.append(i)

        x_o2 = np.stack([np.stack(x_i, axis=0) for x_i in [x_lo, x_mid, x_hi]], axis=0)
        x_input_o2 = np.stack([np.stack(x_i, axis=0) for x_i in [x_input_lo, x_input_mid, x_input_hi]], axis=0)
        print('x_o2:', x_o2.shape)

        flow_o2 = solver(x_input_o2[1:], x_input_o2[:-1], flow_axes=flow_axes,
                         gaussian=gaussian, prefilter=prefilter,
                         radius=(momentum_radius, coop_radius) + group_radius,
                         sigma=(momentum_sigma, coop_sigma) + group_sigma,
                         **flow_kw)

        x_, flow_ = flow.align_to_ref(x_o2, flow_o2, ref_index[ph2_axis], axis=0, return_flows=True)
        # x_input_ = flow.align_to_ref(x_input_o2, flow_o2, ref_index[ph2_axis], axis=0)

        full_flow_o2 = np.zeros((x.shape[po2_axis], x.shape[ph2_axis], np.ndim(x_o2) - 1, *x.shape[num_group_dims:]))
        full_flow_o2.fill(np.nan)
        print(len(flow_), [f.shape if f is not None else 0 for f in flow_], full_flow_o2.shape)
        for stack_i, i in enumerate(o2_index):
            for j in range(x.shape[ph2_axis]):
                if flow_[j] is not None:
                    full_flow_o2[i, j] = flow_[j][stack_i]
                else:
                    full_flow_o2[i, j] = 0

        # Fill in flow for missing po2 values
        print(np.sum(np.isnan(full_flow_o2)))
        full_flow_o2 = impute_nans(full_flow_o2, sigma=(0.5, 0.5, 0) + (0,) * (np.ndim(x) - num_group_dims))

        for j in range(x.shape[ph2_axis]):
            for i in range(x.shape[po2_axis]):
                for k in range(x.shape[ph2o_axis]):
                    ijk = (i, j, k)
                    flows[ijk] += full_flow_o2[i, j][-1]
                    index_nested(flow_sequence, ijk).append(full_flow_o2[i, j, 1:])

                    if group_exists[ijk]:
                        x_align[ijk] = flow.warp(x_align[ijk], full_flow_o2[i, j, 1:])
                        x_input[ijk] = flow.warp(x_input[ijk], full_flow_o2[i, j, 1:])

    else:
        # No cooperative warping along ph2 axis
        x_input_o2 = x_input[2, :, ref_index[ph2o_axis]]
        x_o2 = x_align[2, :, ref_index[ph2o_axis]]

        flow_o2 = solver(x_input_o2[1:], x_input_o2[:-1], flow_axes=flow_axes,
                         gaussian=gaussian, prefilter=prefilter,
                         radius=(momentum_radius,) + group_radius, sigma=(momentum_sigma,) + group_sigma,
                         **flow_kw)

        x_, flow_ = flow.align_to_ref(x_o2, flow_o2, ref_index[ph2_axis], axis=0, return_flows=True)
        x_input_ = flow.align_to_ref(x_input_o2, flow_o2, ref_index[ph2_axis], axis=0)

        for j in range(x.shape[ph2_axis]):
            ijk = (2, j, ref_index[ph2o_axis])
            x_align[ijk] = x_[j]
            x_input[ijk] = x_input_[j]
            if flow_[j] is not None:
                flows[ijk] += flow_[j][-1]
                index_nested(flow_sequence, ijk).append(flow_[j])

            # Apply to off-axis groups
            for i in range(x.shape[po2_axis]):
                for k in range(x.shape[ph2o_axis]):
                    if not (i == 2 and k == ref_index[ph2o_axis]):
                        if flow_[j] is not None:
                            flows[i, j, k] += flow_[j][-1]
                            index_nested(flow_sequence, (i, j, k)).append(flow_[j])
                            if group_exists[i, j, k]:
                                x_group = x_align[i, j, k]
                                x_align[i, j, k] = flow.warp(x_group, flow_[j])
                                x_input[i, j, k] = flow.warp(x_input[i, j, k], flow_[j])

    # Sum flow sequences
    it = np.nditer(x, op_axes=[list(np.arange(num_group_dims))], flags=['multi_index'])
    tot_flow = np.empty((*x.shape[:num_group_dims], 2, *x.shape[num_group_dims:]))
    for group in it:
        group_index = it.multi_index
        fs = index_nested(flow_sequence, group_index)

        if len(fs) > 0:
            # print(group_index)
            # print(tot_flow[group_index].shape)
            # print([f.shape for f in fs])
            tot_flow[group_index] = flow.sum_flows(fs)
        else:
            tot_flow[group_index] = 0

    return x_align, tot_flow, flow_sequence


def warp_groups(x, group_by, tot_flow):
    x_align = x.copy()
    num_group_dims = len(group_by)

    group_exists = ~ndx.group_isnan(x, num_group_dims)

    for group_index in np.argwhere(group_exists):
        index = tuple(group_index)
        x_group = x_align[index]

        # full_flow = np.zeros((np.ndim(x_group), *x_group.shape))
        # full_flow[-1] = flows[index]
        x_align[index] = flow.warp(x_group, tot_flow[index])

    return x_align


def warp_groups_sequential(x, group_by, flow_sequence):
    x_align = x.copy()
    num_group_dims = len(group_by)

    group_exists = ~ndx.group_isnan(x, num_group_dims)

    for group_index in np.argwhere(group_exists):
        index = tuple(group_index)
        x_group = x_align[index]

        fs = index_nested(flow_sequence, index)

        if len(fs) > 0:
            for f in fs:
                x_group = flow.warp(x_group, f)
            x_align[index] = x_group

    return x_align
