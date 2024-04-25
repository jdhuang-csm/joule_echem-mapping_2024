import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import pandas as pd
from scipy import ndimage
import os

import hybdrt.fileload as fl
from hybdrt.utils.eis import complex_vector_to_concat, concat_vector_to_complex, polar_from_complex
from hybdrt.mapping.nddata import flag_bad_obs, flag_outliers, impute_nans

from .dataload import get_pmap_files


def prep_interp_data(iv_df):
    if 'ImExpected' in iv_df.columns:
        i_vals = iv_df['ImExpected'].values
    else:
        i_vals = iv_df['Im'].values

    v_vals = iv_df['Vf'].values

    sort_index = np.argsort(i_vals)
    i_sort = i_vals[sort_index]
    v_sort = v_vals[sort_index]

    # Add points to enable extrapolation near endpoints
    i_range = np.max(i_vals) - np.min(i_vals)
    i_low = np.min(i_vals) - 10 * i_range
    i_high = np.max(i_vals) + 10 * i_range

    if len(i_sort) >= 6:
        # Exclude endpoints due to voltage cutoff
        extrap_points = 4  # number of points to use for extrapolation
        i_sort = i_sort[1:-1]
        v_sort = v_sort[1:-1]
    else:
        # Very few points. Include endpoints
        extrap_points = min(4, len(i_sort))  # number of points to use for extrapolation
    low_fit = np.polyfit(i_sort[:extrap_points], v_sort[:extrap_points], deg=1)
    high_fit = np.polyfit(i_sort[-extrap_points:], v_sort[-extrap_points:], deg=1)
    v_low = np.polyval(low_fit, i_low)
    v_high = np.polyval(high_fit, i_high)
    i_extrap = np.concatenate([[i_low], i_sort, [i_high]])
    v_extrap = np.concatenate([[v_low], v_sort, [v_high]])

    return i_extrap, v_extrap


def get_iv_interp(iv_df):
    i_extrap, v_extrap = prep_interp_data(iv_df)
    iv_interp = interp1d(i_extrap, v_extrap)

    return iv_interp


def get_vi_interp(iv_df):
    i_extrap, v_extrap = prep_interp_data(iv_df)
    vi_interp = interp1d(v_extrap, i_extrap)

    return vi_interp


def check_match_signal(match_signal):
    options = ['i', 'v']
    if match_signal not in options:
        raise ValueError(f'Invalid match value {match_signal}. Options: {options}')


def get_offset_thresh(interp, match_signal, v_mid=None, v_offset=0.005):
    if match_signal == 'i':
        return v_offset
    else:
        return interp(v_mid + v_offset / 2) - interp(v_mid - v_offset / 2)


def test_chrono_iv(chrono_data, interp, match_signal='v', offset_rthresh=0.1, offset_thresh=None,
                   v_offset_thresh=0.005, range_rthresh=0.1):
    check_match_signal(match_signal)

    # Current limits
    i_lo = np.percentile(chrono_data['Im'].values, 2)
    i_hi = np.percentile(chrono_data['Im'].values, 98)

    # Voltage limits
    v_lo = np.percentile(chrono_data['Vf'].values, 2)
    v_hi = np.percentile(chrono_data['Vf'].values, 98)

    # Midpoint current/voltage
    i_mid = 0.5 * (i_lo + i_hi)
    v_mid = 0.5 * (v_lo + v_hi)

    if offset_thresh is None:
        offset_thresh = get_offset_thresh(interp, match_signal, v_mid, v_offset_thresh)
        # print('offset thresh', offset_thresh)

    if match_signal == 'i':
        # Assume recorded current is correct. Check if recorded voltage matches expected voltage
        # Check voltage range
        s_lo_pred = interp(i_lo)
        s_hi_pred = interp(i_hi)
        s_range_pred = (s_hi_pred - s_lo_pred)
        s_range = v_hi - v_lo

        # Check voltage offset
        s_mid_meas = v_mid
        s_mid_pred = interp(i_mid)
        v_oc = interp(0)
        ds_pred = s_mid_pred - v_oc
        ds_meas = v_mid - v_oc

    else:
        # Assume voltage is correct, check current
        # Check current range
        s_lo_pred = interp(v_lo)
        s_hi_pred = interp(v_hi)
        s_range_pred = (s_hi_pred - s_lo_pred)
        s_range = i_hi - i_lo

        # Check current offset
        s_mid_meas = i_mid
        s_mid_pred = interp(v_mid)

        # ds_pred = interp(v_mid)
        # ds_meas = i_mid
        # Ues largest current magnitude
        if abs(i_lo) > abs(i_hi):
            ds_pred = interp(v_lo)
            ds_meas = i_lo
        else:
            ds_pred = interp(v_hi)
            ds_meas = i_hi

    range_factor = s_range_pred / s_range
    # if abs(v_range - v_range_exp) > v_range_thresh:
    #     print('v_range:', v_range, v_range_exp)
    #     return 1

    if abs((ds_meas - ds_pred) / ds_pred) > offset_rthresh and abs(s_mid_meas - s_mid_pred) > offset_thresh:
        if abs(1 - range_factor) > range_rthresh:
            ds_factor = ds_pred / ds_meas
            factor_ratio = (1 - ds_factor) / (1 - range_factor)
            if factor_ratio > 0:
                # if abs(np.log(factor_ratio)) < np.log()
                print('s_mid:', s_mid_meas, s_mid_pred)
                print('range factor:', range_factor)
                print('chrono factor: {:.3f}'.format(ds_factor))
                return 1, ds_factor

    return 0, 1


def test_eis_iv(eis_data, interp, match_signal='v', offset_rthresh=0.1, offset_thresh=None, v_offset_thresh=0.005):
    check_match_signal(match_signal)

    idc = eis_data['Idc'].median()
    vdc = eis_data['Vdc'].median()

    if offset_thresh is None:
        offset_thresh = get_offset_thresh(interp, match_signal, vdc, v_offset_thresh)

    if match_signal == 'i':
        sdc_meas = vdc
        sdc_pred = interp(idc)

        v_oc = interp(0)
        ds_meas = vdc - v_oc
        ds_pred = sdc_pred - v_oc
    else:
        sdc_meas = idc
        sdc_pred = interp(vdc)
        ds_meas = idc
        ds_pred = interp(vdc)

    if abs((ds_meas - ds_pred) / ds_pred) > offset_rthresh and abs(sdc_meas - sdc_pred) > offset_thresh:
        print('EIS factor: {:.3f}'.format(ds_pred / ds_meas))
        print('')
        return 1, ds_pred / ds_meas
    else:
        return 0, 1


def correct_chrono(data, factor, interp, match_signal='v'):
    cor_data = data.copy()

    if match_signal == 'i':
        v_oc = interp(0)
        cor_data['Vf'] = v_oc + (cor_data['Vf'] - v_oc) * factor
    elif match_signal == 'v':
        cor_data['Im'] = cor_data['Im'] * factor

    return cor_data


def correct_eis(data, factor, interp, match_signal='v'):
    check_match_signal(match_signal)
    cor_data = data.copy()

    if match_signal == 'v':
        cor_data['Idc'] = cor_data['Idc'] * factor
        z_factor = 1 / factor
    else:
        v_oc = interp(0)
        cor_data['Vdc'] = v_oc + (cor_data['Vdc'] - v_oc) * factor
        z_factor = factor

    for col in ['Zreal', 'Zimag', 'Zmod']:
        cor_data[col] = cor_data[col] * z_factor

    return cor_data


def read_header(file):
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()

    # find start of curve data
    table_index = txt.upper().find('CURVE\tTABLE')

    # Include column names and units
    end_index = table_index + len('\n'.join(txt[table_index:].split('\n')[:3]))

    return txt[:end_index + 1]


def correct_step_pmap_data(data_path, step_id, mode_match_signal=None, filtered=False,
                           data_type='hybrid', include_full_eis=True, impute_bad_z=True,
                           offset_rthresh=0.1, offset_thresh=None, v_offset_thresh=0.005,
                           range_rthresh=0.1,
                           write_corrected=False, clear_corrected=False):
    if mode_match_signal is None:
        mode_match_signal = {'discharge': 'v', 'charge': 'v'}

    for match_signal in mode_match_signal.values():
        check_match_signal(match_signal)

    data_path = Path(data_path)

    # Load IV file
    pwr_file = list(data_path.glob(f'PWRPOLARIZATION_Step{step_id}*.DTA'))
    if len(pwr_file) > 0:
        iv_file = pwr_file[0]
        iv_df = fl.read_curve(iv_file)
    else:
        iv_file = list(data_path.glob(f'VSWEEPIV_Step{step_id}*.DTA'))[0]
        iv_df = pd.read_csv(iv_file, sep='\t')

    # iv_file = Path(iv_file)
    # if iv_file.name.find('PWRPOL') > -1:
    #     iv_df = fl.read_curve(iv_file)
    # else:
    #     iv_df = pd.read_csv(iv_file)

    # Make IV interpolants
    iv_interp = get_iv_interp(iv_df)
    vi_interp = get_vi_interp(iv_df)

    mode_results = {}
    for n, (mode, match_signal) in enumerate(mode_match_signal.items()):
        print(f'Mode: {mode}')
        # Include OCV only for the first mode to avoid duplication
        include_ocv = (n == 0)

        # Get pmap files
        chrono_files, eis_files = get_pmap_files(data_path, step_id, data_type=data_type,
                                                 mode=mode, include_full_eis=include_full_eis,
                                                 include_ocv=include_ocv,
                                                 prefer_corrected=False, prefer_filtered=filtered)

        if clear_corrected:
            for file in chrono_files + eis_files:
                if file is not None:
                    corrected_file = os.path.join(file.parent, file.name + '-CORRECTED')
                    if os.path.exists(corrected_file):
                        os.remove(corrected_file)
                        print('Deleted previously corrected file {}'.format(os.path.basename(corrected_file)))

        # Make IV interpolant
        if match_signal == 'i':
            interp = iv_interp
        else:
            interp = vi_interp

        # Initialize arrays/lists
        chrono_dfs = []
        eis_dfs = []
        i_array = []
        v_array = []
        z_array = []
        signal_vals = np.empty(len(chrono_files))

        chrono_flags = np.zeros(len(chrono_files))
        eis_flags = np.zeros(len(chrono_files))
        chrono_factors = np.ones(len(chrono_files))
        eis_factors = np.ones(len(chrono_files))

        # Load data and check IV
        for i in range(len(chrono_files)):
            chrono_file = chrono_files[i]
            eis_file = eis_files[i]

            if chrono_file is not None:
                chrono_df = fl.read_chrono(chrono_file)
                chrono_dfs.append(chrono_df)
                i_array.append(chrono_df['Im'].values)
                v_array.append(chrono_df['Vf'].values)

                chrono_flags[i], chrono_factors[i] = test_chrono_iv(chrono_df, interp, match_signal,
                                                                    offset_rthresh, offset_thresh, v_offset_thresh,
                                                                    range_rthresh)

                # Get midpoint signal
                if match_signal == 'i':
                    s_chrono = chrono_df['Im'].values
                else:
                    s_chrono = chrono_df['Vf'].values
                s_mid = 0.5 * (np.percentile(s_chrono, 98) + np.percentile(s_chrono, 2))
            else:
                chrono_dfs.append(None)
                i_array.append(None)
                v_array.append(None)
                s_mid = None

            if eis_file is not None:
                eis_df = fl.read_eis(eis_file)
                eis_dfs.append(eis_df)
                _, z = fl.get_eis_tuple(eis_df)
                z_array.append(z)

                # EIS factor is difficult to verify on its own. Only test EIS data if chrono data raised a flag
                if chrono_flags[i] == 1:
                    eis_flags[i], eis_factors[i] = test_eis_iv(eis_df, interp, match_signal, offset_rthresh,
                                                               offset_thresh, v_offset_thresh)

                # Get DC current
                if match_signal == 'i':
                    s_dc = eis_df['Idc'].median()
                else:
                    s_dc = eis_df['Vdc'].median()
            else:
                eis_dfs.append(None)
                z_array.append(None)
                s_dc = None

            if s_mid is not None:
                signal_vals[i] = s_mid
            else:
                signal_vals[i] = s_dc

        # Format arrays
        # chrono data: all should be same length
        v_len = np.array([0 if v is None else len(v) for v in v_array])
        # print(v_len)
        v_len = np.unique(v_len[v_len > 0])
        if len(v_len) > 1:
            raise ValueError(f'Found chrono data with different lengths: {v_len}')
        else:
            v_len = v_len[0]
        i_array = np.stack([np.empty(v_len) * np.nan if im is None else im for im in i_array], axis=0)
        v_array = np.stack([np.empty(v_len) * np.nan if v is None else v for v in v_array], axis=0)

        # EIS data: truncate to shortest length (hybrid measurements)
        z_len = np.min([np.inf if z is None else len(z) for z in z_array])
        z_array = np.stack(
            [np.empty(2 * z_len) * np.nan if z is None else complex_vector_to_concat(z[:z_len]) for z in z_array],
            axis=0
        )

        # Sort by match signal
        sort_index = np.argsort(signal_vals)
        chrono_dfs = [chrono_dfs[i] for i in sort_index]
        eis_dfs = [eis_dfs[i] for i in sort_index]
        chrono_files = [chrono_files[i] for i in sort_index]
        eis_files = [eis_files[i] for i in sort_index]
        i_array = i_array[sort_index]
        v_array = v_array[sort_index]
        z_array = z_array[sort_index]
        chrono_factors = chrono_factors[sort_index]
        eis_factors = eis_factors[sort_index]
        chrono_flags = chrono_flags[sort_index]
        eis_flags = eis_flags[sort_index]
        signal_vals = signal_vals[sort_index]

        # Apply factors
        if match_signal == 'i':
            i_clean = i_array.copy()
            v_oc = interp(0)
            v_clean = v_oc + (v_array - v_oc) * chrono_factors[:, None]
            z_clean = z_array * eis_factors[:, None]
        else:
            i_clean = i_array * chrono_factors[:, None]
            v_clean = v_array.copy()
            z_clean = z_array / eis_factors[:, None]

        mode_results[mode] = {
            'chrono_dfs': chrono_dfs,
            'eis_dfs': eis_dfs,
            'chrono_files': chrono_files,
            'eis_files': eis_files,
            'i_array': i_array,
            'v_array': v_array,
            'z_array': z_array,
            'i_clean': i_clean,
            'v_clean': v_clean,
            'z_clean': z_clean,
            'chrono_factors': chrono_factors,
            'eis_factors': eis_factors,
            'chrono_flags': chrono_flags,
            'eis_flags': eis_flags,
        }

    # Append mode results in order
    modes = ['discharge', 'charge']
    mode_results = {k: mode_results[k] for k in modes if k in mode_results.keys()}
    modes = list(mode_results.keys())
    results = {}
    for key in ['chrono_dfs', 'eis_dfs', 'chrono_files', 'eis_files']:
        results[key] = sum([mode_results[mode][key] for mode in modes], [])

    for key in ['i_array', 'v_array', 'z_array',
                'i_clean', 'v_clean', 'z_clean',
                'chrono_factors', 'eis_factors', 'chrono_flags', 'eis_flags']:
        results[key] = np.concatenate([mode_results[mode][key] for mode in modes], axis=0)

    match_signal_list = sum(
        [[mode_match_signal[k]] * len(mode_results[k]['chrono_files']) for k in mode_results.keys()],
        []
    )
    results['match_signal'] = match_signal_list

    i_clean = results['i_clean']
    v_clean = results['v_clean']
    z_clean = results['z_clean']

    # Get v_diff for comparison
    v_hi = np.nanpercentile(v_clean, 98, axis=1)
    v_lo = np.nanpercentile(v_clean, 2, axis=1)
    v_mid = 0.5 * (v_hi + v_lo)
    i_range = np.nanpercentile(i_clean, 98, axis=1) - np.nanpercentile(i_clean, 2, axis=1)
    v_diff = (v_clean - v_mid[:, None]) / i_range[:, None]

    # Flag individual outlier points
    v_out_flag = flag_outliers(v_diff, filter_size=(5, 1), thresh=0.7)
    z_out_flag = flag_outliers(z_clean, filter_size=(5, 3), thresh=0.7)
    print('z_out:', np.where(z_out_flag))

    # If fewer than 5% of data points in any observation are outliers,
    # set outliers to nan prior to checking for bad observations.
    v_out_count = np.sum(v_out_flag, axis=1)
    v_count_index = v_out_count < int(v_diff.shape[1] * 0.05)
    v_diff[v_count_index[:, None] & v_out_flag] = np.nan

    z_out_count = np.sum(z_out_flag, axis=1)
    z_count_index = z_out_count < int(z_clean.shape[1] * 0.05)
    z_clean[z_count_index[:, None] & z_out_flag] = np.nan

    # # Flag observations with outliers
    # chrono_flags[v_count_index & (v_out_count > 0)] = 1
    # eis_flags[z_count_index * (z_out_count > 0)] = 1

    # Find any remaining uncorrected EIS files that may have been missed (especially pure EIS measurements)
    # DON'T test further factor corrections to v - if v_diff is off by a factor but was not corrected above,
    # it is unclear what the issue is and the measurement should be excluded
    for i in range(5):
        # Repeat to reassess after bad obs are fixed
        z_filt = ndimage.median_filter(z_clean, size=(5, 1))

        # (v_bad, z_bad), (v_diff_cor, z_cor) = flag_bad_obs((v_diff, z_clean), (v_filt, z_filt),
        #                                                    thresh=1, std_size=5, test_factor_correction=True)

        z_bad, z_cor, z_rss = flag_bad_obs(z_clean, z_filt, thresh=0.5, std_size=(5, 3), test_factor_correction=True,
                                           return_rss=True)

        # Find changed observations
        # v_cor = v_diff_cor * i_range[:, None] + v_mid[:, None]
        # v_change = np.any(np.round(np.nan_to_num(v_cor), 6) != np.round(np.nan_to_num(v_clean), 6), axis=1)
        # chrono_flags[v_change] = 1
        # v_clean = v_cor

        z_change = np.any(np.round(np.nan_to_num(z_cor), 6) != np.round(np.nan_to_num(z_clean), 6), axis=1)
        results['eis_flags'][z_change] = 1
        z_change_factors = np.median(z_cor[z_change] / z_clean[z_change], axis=1)
        # print(z_change_factors)
        z_factor_exponents = np.array([1 if ms == 'i' else -1 for ms in match_signal_list])[z_change]
        # print(z_change_factors.shape, z_factor_exponents.shape, results['eis_factors'][z_change].shape)
        results['eis_factors'][z_change] *= z_change_factors ** z_factor_exponents
        z_clean = z_cor

        if not np.any(z_change):
            # If no observations changed, don't need to repeat
            break

    results['z_clean'] = z_clean

    # Flag bad chrono observations without testing corrections
    v_filt = ndimage.median_filter(v_diff, size=(3, 1))
    v_bad, v_rss = flag_bad_obs(v_diff, v_filt, thresh=0.5, std_size=(5, 3), test_factor_correction=False,
                                return_rss=True)

    # z_ratio = np.median(z_filt / z_clean, axis=1)
    # ratio_flag = np.where(np.abs(np.log(z_ratio)) > np.log(z_ratio_thresh))[0]
    # if len(ratio_flag) > 0:
    #     eis_flags[ratio_flag] = 1
    #     # print(z_ratio[ratio_flag], np.shape(z_ratio[ratio_flag]))
    #     eis_factors[ratio_flag] *= z_ratio[ratio_flag]
    #     z_clean[ratio_flag] *= z_ratio[ratio_flag][:, None]

    if impute_bad_z:
        # Re-check z outliers after factor corrections
        z_tmp = z_clean.copy()
        z_tmp[z_bad] = np.nan
        z_tmp = impute_nans(z_tmp, sigma=(0.5, 0))
        z_out_flag = flag_outliers(z_tmp, filter_size=(5, 1), thresh=0.7)
        print('z_out:', np.where(z_out_flag))

        z_imp = z_clean.copy()
        z_imp[z_bad] = np.nan
        z_imp[z_out_flag] = np.nan
        eis_impute_flags = np.any(np.isnan(z_imp), axis=1)
        z_imp = impute_nans(z_imp, sigma=(0.5, 0))

        # Don't impute pure EIS measurements - can't impute low freq data
        is_pure_eis = np.all(np.isnan(v_clean), axis=1)
        z_imp[is_pure_eis] = z_clean[is_pure_eis]
        eis_impute_flags[is_pure_eis] = 0

        # Perform final check for bad observations after imputation
        z_filt = ndimage.median_filter(z_imp, size=(5, 1))
        z_bad, z_rss = flag_bad_obs(z_imp, z_filt, thresh=0.5, std_size=(5, 3), test_factor_correction=False,
                                    return_rss=True)

        # Flag imputed measurements as corrected
        results['eis_flags'][eis_impute_flags] = 1

        results['z_clean'] = z_imp
    else:
        eis_impute_flags = np.zeros(len(z_clean), dtype=bool)
        z_imp = None

    results['eis_impute_flags'] = eis_impute_flags

    print('v_bad:', np.where(np.any(v_bad, axis=1)))
    print('z_bad:', np.where(np.any(z_bad, axis=1)))
    results['v_bad'] = v_bad
    results['z_bad'] = z_bad

    if write_corrected:
        def write_cor(original_file, corrected_df):
            new_file = Path(os.path.join(original_file.parent, original_file.name + '-CORRECTED'))

            header = read_header(original_file)
            columns = [c for c in corrected_df.columns if c != 'timestamp']
            data_txt = corrected_df.to_csv(None, sep='\t', header=False, line_terminator='\n',
                                           index=False, columns=columns,
                                           float_format=f'%.6g')
            # Block indent
            data_txt = '\t' + data_txt.replace('\n', '\n\t')[:-1]

            with new_file.open('w+') as f:
                f.write(header)
                f.write(data_txt)

            print(f'Wrote corrected data to {new_file.name}')
            return new_file

        for i in np.where(results['chrono_flags'] == 1)[0]:
            # new_df = chrono_dfs[i].copy()
            if match_signal_list[i] == 'i':
                interp = iv_interp
            else:
                interp = vi_interp

            new_df = correct_chrono(results['chrono_dfs'][i], results['chrono_factors'][i],
                                    interp, match_signal_list[i])

            results['chrono_files'][i] = write_cor(results['chrono_files'][i], new_df)

        for i in np.where(results['eis_flags'] == 1)[0]:
            if match_signal_list[i] == 'i':
                interp = iv_interp
            else:
                interp = vi_interp

            new_df = correct_eis(results['eis_dfs'][i], results['eis_factors'][i], interp, match_signal_list[i])

            if impute_bad_z:
                # Overwrite impedance directly with imputed data
                # Still need to apply any correction factors above to set Idc/Vdc correctly
                if eis_impute_flags[i]:
                    z_new = concat_vector_to_complex(z_imp[i])
                    new_df['Zreal'] = z_new.real
                    new_df['Zimag'] = z_new.imag
                    zmod, zphz = polar_from_complex(z_new)
                    new_df['Zmod'] = zmod
                    new_df['Zphz'] = zphz

            results['eis_files'][i] = write_cor(results['eis_files'][i], new_df)

    # Store results in csv
    fname_array = np.array([[f if f is None else f.name for f in results[name]]
                            for name in ['eis_files', 'chrono_files']])
    res_df = pd.DataFrame(fname_array.T, columns=['eis_file', 'chrono_file'])
    res_df['chrono_corrected'] = results['chrono_flags']
    res_df['chrono_factor'] = results['chrono_factors']
    res_df['v_rss'] = v_rss
    res_df['eis_corrected'] = results['eis_flags']
    res_df['eis_factor'] = results['eis_factors']
    res_df['z_imputed'] = results['eis_impute_flags']
    res_df['z_rss'] = z_rss
    res_df.to_csv(data_path.joinpath(f'DataSummary_Step{step_id}.csv'), index_label='Index')

    return results


def correct_path_pmap_data(data_path, mode_match_signal=None, start_step_num=0, filtered=False, data_type='hybrid',
                           include_full_eis=True, impute_bad_z=True,
                           offset_rthresh=0.1, offset_thresh=None, v_offset_thresh=0.005, range_rthresh=0.1,
                           write_corrected=False, clear_corrected=False):
    data_path = Path(data_path)

    chrono_corrected = 0
    eis_corrected = 0

    while True:
        step_id = f'{start_step_num}d'

        if len(list(data_path.glob(f'*Step{step_id}*.DTA'))) > 0:
            print(f'Checking step {step_id}...')
            results = correct_step_pmap_data(data_path, step_id, mode_match_signal, filtered,
                                             data_type, include_full_eis, impute_bad_z=impute_bad_z,
                                             offset_rthresh=offset_rthresh, offset_thresh=offset_thresh,
                                             v_offset_thresh=v_offset_thresh,
                                             range_rthresh=range_rthresh,
                                             write_corrected=write_corrected,
                                             clear_corrected=clear_corrected)
            chrono_corrected += np.sum(results['chrono_flags'])
            eis_corrected += np.sum(results['eis_flags'])

            start_step_num += 1
        else:
            break

    print(f'Corrected a total of {chrono_corrected} chrono files and {eis_corrected} EIS files in path {data_path}')
