from pathlib import Path
import numpy as np

import hybdrt.fileload as fl

from . import sequence, capacity
from .reader import get_key, ReaderCollection
from .utils import get_cycle, get_mode, get_test_id

import shutil
import os

def get_test_data(testdir, mode, filtered=True, max_num_obs=100, include_v_finish=False):
    testdir = Path(testdir)

    # Get chrono files
    if filtered:
        chrono_pattern = f'CHRONOP_{mode.upper()}*_Cycle*_Filtered.DTA'
    else:
        chrono_pattern = f'CHRONOP_{mode.upper()}*_Cycle*.DTA'
    chrono_files = list(testdir.glob(chrono_pattern))
    chrono_files = sorted(chrono_files, key=get_cycle)  # [:2]
    print(f'Found {len(chrono_files)} files for mode {mode.upper()}')

    # Get EIS files
    eis_files = list(testdir.glob(f'EISGALV_{mode.upper()}*_Cycle*.DTA'))
    eis_files = sorted(eis_files, key=get_cycle)  # [:2]

    # Check number of files
    if len(chrono_files) != len(eis_files):
        raise ValueError(f'Number of chrono files ({len(chrono_files)}) '
                         f'is different than the number of EIS files ({len(eis_files)})')

    # Get conditioning file
    cond_file = next(testdir.glob(f'Conditioning_{mode.upper()}*.DTA'))

    # Get voltage finish file
    finish_file = None
    if include_v_finish:
        finish_files = list(testdir.glob(f'PSTATIC-FINISH_{mode.upper()}*.DTA'))
        if len(finish_files) > 0:
            finish_file = finish_files[0]

    # Cap total number of observations
    num_files = len(chrono_files)
    if num_files > max_num_obs:
        # SKip first file after conditioning - often seems to exhibit slight non-ideality
        file_index = np.round(np.linspace(1, num_files - 1, max_num_obs)).astype(int)
        chrono_files = [chrono_files[i] for i in file_index]
        eis_files = [eis_files[i] for i in file_index]
        print(f'Downsampled to {len(chrono_files)} files')
        
    # Concatenate IVT data
    # Keep the individual dataframes for psi calculation in load_test_data to avoid multiple file reads
    chrono_dfs = [fl.read_chrono(file) for file in chrono_files]
    eis_dfs = [fl.read_eis(file) for file in eis_files]
    # Include conditioning file in sequence data, but exclude from observation files
    chrono_seq_df = fl.concatenate_chrono_data([fl.read_chrono(cond_file)] + chrono_dfs)
    eis_seq_df = fl.concatenate_eis_data(eis_dfs)
    # chrono_seq_df = fl.concatenate_chrono_data([cond_file] + chrono_files, loop=True, print_progress=True)
    # eis_seq_df = fl.concatenate_eis_data(eis_files, loop=True, print_progress=True)

    # Append voltage finish
    if finish_file is not None:
        chrono_seq_df = fl.concatenate_chrono_data([chrono_seq_df, fl.read_chrono(finish_file)])

    # Get timestamp corresponding to start of current application
    init_timestamp = chrono_seq_df[np.abs(chrono_seq_df['Im']) > 0.01].reset_index().loc[0, 'timestamp']
    # init_timestamp = chrono_seq_df.loc[0, 'timestamp']

    # Align EIS and chrono times to initial timestamp
    chrono_seq_df['elapsed'] = (chrono_seq_df['timestamp'] - init_timestamp).dt.total_seconds()
    eis_seq_df['elapsed'] = (eis_seq_df['timestamp'] - init_timestamp).dt.total_seconds()
    # eis_seq_df['elapsed'] += fl.get_time_offset(eis_seq_df, chrono_seq_df)

    # Get chrono data at DC (center) current
    chrono_mid_df = chrono_seq_df.copy()
    chrono_mid_df = chrono_mid_df[
        (chrono_mid_df['Time'] < 1) | (chrono_mid_df['elapsed'] < 30) |
        # Include voltage finish
        (chrono_mid_df['timestamp'] > chrono_dfs[-1]['timestamp'].values[-1])
    ]

    # Merge chrono and EIS data
    t_chrono, i_chrono, v_chrono = fl.get_chrono_tuple(chrono_mid_df)
    t_eis, i_eis, v_eis = fl.iv_from_eis(eis_seq_df)
    times, unique_index = np.unique(np.concatenate([t_chrono, t_eis]), return_index=True)
    i_sig = np.concatenate([i_chrono, i_eis])[unique_index]
    v_sig = np.concatenate([v_chrono, v_eis])[unique_index]
    cp_seq_tup = (times, i_sig, v_sig)

    # Get DC current and total duration
    i_dc = np.median(i_sig[times >= 0])
    tot_time = chrono_seq_df['elapsed'].values[-1]
    # t_dc = times[np.abs(i_sig) > 0.01][0]

    return chrono_files, eis_files, chrono_dfs, eis_dfs, init_timestamp, i_dc, tot_time, cp_seq_tup


def load_test_data(drtmd, datadir, test_id, tot_capacity, start_timestamp, filtered=True, seq_downsample_size=1000,
                   max_num_obs=100, trust_charge_finish=True):
    print(f'Loading data for Test {test_id}...')
    datadir = Path(datadir)
    testdir = next(datadir.glob(f'Test{test_id}_*'))
    print(testdir.name)

    mode_data = {}
    for mode in ['charge', 'discharge']:
        mode_data[mode] = {}

        # Load data
        chrono_files, eis_files, chrono_dfs, eis_dfs, init_timestamp, i_dc, duration, cp_seq_tup = \
            get_test_data(testdir, mode, filtered=filtered, max_num_obs=max_num_obs)

        mode_data[mode]['chrono_files'] = chrono_files
        mode_data[mode]['eis_files'] = eis_files
        mode_data[mode]['chrono_dfs'] = chrono_dfs
        mode_data[mode]['eis_dfs'] = eis_dfs
        mode_data[mode]['init_timestamp'] = init_timestamp
        mode_data[mode]['i_dc'] = i_dc
        mode_data[mode]['duration'] = duration
        mode_data[mode]['cp_seq_tup'] = cp_seq_tup

        # Calculate the nominal capacity change
        cap_delta = capacity.calc_capacity_delta(duration, i_dc)

        # Add voltage finish capacity
        finish_files = list(testdir.glob(f'PSTATIC-FINISH_{mode.upper()}*.DTA'))
        if len(finish_files) >= 1:
            for file in finish_files:
                cap_delta += capacity.integrate_pstatic_capacity(file)
        # elif len(finish_files) > 1:
        #     raise ValueError('Found multiple voltage finish files:', finish_files)

        mode_data[mode]['cap_delta'] = cap_delta

        # Fit the midpoint voltage for chrono offset
        mode_data[mode]['drt_seq'] = sequence.fit_sequence_data(cp_seq_tup, downsample_size=seq_downsample_size)

    if trust_charge_finish:
        # Get effective discharge factor from ratio of charge and discharge capacities
        discharge_factor = -mode_data['charge']['cap_delta'] / mode_data['discharge']['cap_delta']
        print('Charge capacity: {:.0f} mAh'.format(mode_data['charge']['cap_delta']))
        print('Discharge capacity: {:.0f} mAh'.format(mode_data['discharge']['cap_delta']))
        print('Discharge factor: {:.3f}'.format(discharge_factor))
        mode_data['charge']['i_factor'] = 1
        mode_data['discharge']['i_factor'] = discharge_factor

        mode_data['charge']['init_soc'] = 1 - mode_data['charge']['cap_delta'] / tot_capacity
        mode_data['discharge']['init_soc'] = 1
    else:
        # Charge finish file is incorrect - use discharged capacity to determine initial SOC for charge
        mode_data['charge']['i_factor'] = 1
        mode_data['discharge']['i_factor'] = 1

        mode_data['charge']['init_soc'] = 1 + mode_data['discharge']['cap_delta'] / tot_capacity
        mode_data['discharge']['init_soc'] = 1

    for mode in mode_data.keys():
        data = mode_data[mode]

        reader_key = f'{test_id}_{mode.upper()}'

        # Make the chrono reader function
        chrono_reader = sequence.make_chrono_reader(data['init_timestamp'], data['drt_seq'])

        # Add it to the reader collection
        drtmd.chrono_reader.add_reader(reader_key, chrono_reader)

        # Get C rate
        c_rate = round(data['i_dc'] * 1000 / tot_capacity, 1)

        # Load files

        # # Use a uniformly spaced subset of files to cap total number of observations
        # num_files = len(chrono_files)
        # if num_files > max_num_obs:
        #     file_index = np.round(np.linspace(0, num_files - 1, max_num_obs)).astype(int)
        # else:
        #     file_index = np.arange(num_files)

        for i in range(len(data['chrono_files'])):
            chrono_file = data['chrono_files'][i]
            eis_file = data['eis_files'][i]
            chrono_df = data['chrono_dfs'][i]
            eis_df = data['eis_dfs'][i]

            # chrono_df = fl.read_chrono(chrono_file)
            # eis_df = fl.read_eis(eis_file)

            # Get midpoint SOC
            soc = capacity.get_hybrid_soc(chrono_df, eis_df, data['init_timestamp'],
                                          data['i_dc'] * data['i_factor'], tot_capacity,
                                          data['init_soc'])

            time_under_load = (eis_df.loc[0, 'timestamp'] - data['init_timestamp']).total_seconds() / 60
            elapsed_time = (eis_df.loc[0, 'timestamp'] - start_timestamp).total_seconds() / 60

            psi = [c_rate, soc, time_under_load, elapsed_time]

            drtmd.add_observation(psi, chrono_file, eis_file, group_id=reader_key)

    return mode_data












