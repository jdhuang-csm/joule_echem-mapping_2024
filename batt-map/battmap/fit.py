import numpy as np
from pathlib import Path
import pickle

from .dataload import load_test_data


def fit_test_data(drtmd, datadir, fit_path, test_id, tot_capacity, start_timestamp, suffix='',
                  filtered=True, seq_downsample_size=1000,
                  max_num_obs=100, trust_charge_finish=True, resolve_kw=None,
                  fit_discharge=True, fit_charge=True):
    print(f'Test {test_id}\n=============')
    fit_path = Path(fit_path)

    # Clear observations
    drtmd.clear_obs()

    # Load test data
    test_data = load_test_data(drtmd, datadir, test_id, tot_capacity, start_timestamp, filtered=filtered,
                   seq_downsample_size=seq_downsample_size, max_num_obs=max_num_obs,
                   trust_charge_finish=trust_charge_finish)
    
    # Store test data for reference
    # First delete unnecessary items to reduce file size
    for mode in ['charge', 'discharge']:
        for key in ['chrono_files', 'eis_files', 'chrono_dfs', 'eis_dfs']:
            del test_data[mode][key]
    with open(fit_path.joinpath(f'CycleData_Test{test_id}{suffix}.pkl'), 'wb') as f:
        pickle.dump(test_data, f, pickle.DEFAULT_PROTOCOL)

    # Fit all observations
    if fit_discharge:
        print('Fitting discharge data...')
        discharge_index = drtmd.get_group_index(f'{test_id}_DISCHARGE')
        drtmd.fit_observations(discharge_index)
    if fit_charge:
        print('Fitting discharge data...')
        charge_index = drtmd.get_group_index(f'{test_id}_CHARGE')
        drtmd.fit_observations(charge_index)

    # Resolve groups
    if resolve_kw is None:
        resolve_kw = dict(psi_sort_dims=['soc'],
                          sigma=0.75, lambda_psi=140, batch_size=7)

    for group in np.unique(drtmd.obs_group_id):
        drtmd.resolve_group(group, **resolve_kw)

    # Save fits
    drtmd.save_attributes('all', fit_path.joinpath(f'Fits_Test{test_id}{suffix}.pkl'))
