import numpy as np
from scipy.integrate import cumtrapz
import pickle

from . import sequence, capacity
from . import dataload as dl



def load_pickle(src):
    with open(src, 'rb') as f:
        out = pickle.load(f)
    return out


def process_discharge_curves(datadir, tot_cap, pickledir=None, suffix='', max_num_obs=50):

    test_data = {}
    for test_id in range(1, 16):
        print(f'Test {test_id}')
        testdir = next(datadir.glob(f'Test{test_id}*'))
        test_data[test_id] = {}
        if pickledir is not None:
            # Load preprocessed data from pickle
            pickled = load_pickle(pickledir.joinpath(f'CycleData_Test{test_id}{suffix}.pkl'))
            # Only keep what we need
            for mode in ['charge', 'discharge']:
                test_data[test_id][mode] = {}
                for k in ['i_dc', 'cp_seq_tup', 'init_timestamp', 'drt_seq']:
                    test_data[test_id][mode][k] = pickled[mode][k]
                i_dc = pickled[mode]['i_dc']
                c_rate = abs(round(i_dc * 1000 / tot_cap, 1))
                test_data[test_id][mode]['c_rate'] = c_rate
                
                if mode == 'charge':
                    finish_file = next(testdir.glob(f'PSTATIC-FINISH_{mode.upper()}*.DTA'))
                    finish_cap = capacity.integrate_pstatic_capacity(finish_file)
                    test_data[test_id][mode]['finish_cap'] = finish_cap
        else:
            # Load raw data files
            for mode in ['charge', 'discharge']:
                chrono_files, eis_files, chrono_dfs, eis_dfs, init_timestamp, i_dc, tot_time, cp_seq_tup = \
                    dl.get_test_data(testdir, mode, max_num_obs=max_num_obs, include_v_finish=False)
                c_rate = round(i_dc * 1000 / tot_cap, 1)
                test_data[test_id][mode] = {'cp_seq_tup': cp_seq_tup,
                                            'i_dc': i_dc,
                                            'c_rate': abs(c_rate),
                                            'init_timestamp': init_timestamp}

                if mode == 'charge':
                    finish_file = next(testdir.glob(f'PSTATIC-FINISH_{mode.upper()}*.DTA'))
                    finish_cap = capacity.integrate_pstatic_capacity(finish_file)
                    test_data[test_id][mode]['finish_cap'] = finish_cap

    # Fit DRT to charge/discharge curves and generate model points for differential curves
    for test_id, data in test_data.items():
        for i, mode in enumerate(['discharge']):  # , 'charge']):
            seq_tup = data[mode]['cp_seq_tup']
            if 'drt_seq' in data[mode].keys():
                drt_seq = test_data[test_id][mode]['drt_seq']
            else:
                drt_seq = sequence.fit_sequence_data(seq_tup, downsample_size=1000)
                test_data[test_id][mode]['drt_seq'] = drt_seq

            t_meas = seq_tup[0]
            t_range = t_meas[-1] - t_meas[0]

            #         t_pred = np.linspace(seq_tup[0][0], seq_tup[0][-1], 10000)
            t_pred1 = np.linspace(t_meas[0], t_meas[0] + t_range * 0.1, 500)
            t_pred2 = np.linspace(t_meas[0] + t_range * 0.1, t_meas[-1], 1000)
            t_pred = np.unique(np.concatenate((t_pred1, t_pred2)))
            v_pred = drt_seq.predict_response(t_pred, subtract_background=False)
            test_data[test_id][mode]['pred_tup'] = (t_pred, np.ones(len(t_pred)) * data[mode]['i_dc'], v_pred)

    # Calculate charge/discharge curves
    for test_id, data in test_data.items():
        for i, mode in enumerate(['discharge']):  # , 'charge']):

            for tup_key, prefix in zip(['cp_seq_tup', 'pred_tup'], ['', 'pred_']):
                times, i_sig, v_sig = data[mode][tup_key]

                #         cap = np.zeros(len(times))
                #         cap[times > 0] = times[times > 0] * abs(i_dc)
                cap_delta = 1000 * cumtrapz(i_sig, x=times) / 3600
                cap_delta = np.insert(cap_delta, 0, 0)
                data[mode][f'{prefix}cap_delta'] = cap_delta
                if mode == 'charge':
                    init_cap = tot_cap - (cap_delta[-1] + data[mode]['finish_cap'])
                    #             init_cap = tot_cap + data['discharge']['cap_delta'][-1]
                    plot_cap = init_cap + cap_delta
                else:
                    init_cap = tot_cap
                    plot_cap = np.abs(cap_delta)

                data[mode][f'{prefix}init_cap'] = init_cap
                data[mode][f'{prefix}plot_cap'] = plot_cap

                cap = init_cap + cap_delta
                data[mode][f'{prefix}cap'] = cap
                data[mode][f'{prefix}soc'] = cap / tot_cap

    return test_data



