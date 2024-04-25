from pathlib import Path
import numpy as np

import hybdrt.fileload as fl
from hybdrt import mapping

from .dataload import add_pmap_obs


def fit_step_data(drtmd, data_path, step_id, init_time, area,
                  include_full_eis=True, prefer_corrected=True, prefer_filtered=False):

    psi_fields = ['Temperature', 'NegatrodeFracH2', 'PositrodeFracO2', 'PositrodePH2O']

    add_pmap_obs(drtmd, data_path, step_id, psi_fields, init_time, area=area,
                 include_full_eis=include_full_eis,
                 prefer_corrected=prefer_corrected,
                 prefer_filtered=prefer_filtered
                 )

    drtmd.fit_all(ignore_errors=True, refit=False)


def fit_path_data(drtmd, data_path, fit_path, prefix, init_time, area, step_range=None, split_by_temp=True,
                  group_id_suffix='',
                  include_full_eis=True, prefer_corrected=True, prefer_filtered=False,
                  score_badness=True, resolve=True, resolve_kw=None
                  ):

    data_path = Path(data_path)
    fit_path = Path(fit_path)

    psi_fields = ['Temperature', 'NegatrodeFracH2', 'PositrodeFracO2', 'PositrodePH2O']

    if step_range is None:
        step_range = (0, 100)

    temp = None
    file_start_step = step_range[0]

    def get_fname():
        if split_by_temp:
            add_str = f'_T={int(temp)}'
        else:
            add_str = ''

        if step_num - file_start_step == 1:
            name = f'{prefix}_Step{file_start_step}{add_str}.pkl'
        else:
            name = f'{prefix}_Steps{file_start_step}-{step_num - 1}{add_str}.pkl'
        return name

    for step_num in range(*step_range):
        step_id = f'{step_num}d'

        step_files = list(data_path.glob(f'*Step{step_id}*.DTA'))

        if len(step_files) > 0:
            # Check temperature
            notes = fl.read_notes(step_files[0], parse=True)
            new_temp = float(notes['Temperature'])
            if temp is None:
                temp = new_temp

            if new_temp != temp and split_by_temp:
                # All steps fitted for current temperature
                # Save fits and clear observations prior to starting next temperature
                fname = get_fname()
                drtmd.save_attributes('all', fit_path.joinpath(fname))
                print(f'Saved {fname}')
                drtmd.clear_obs()

                file_start_step = step_num
                temp = new_temp

            add_pmap_obs(drtmd, data_path, step_id, psi_fields, init_time, area=area,
                         group_id_suffix=group_id_suffix,
                         include_full_eis=include_full_eis,
                         prefer_corrected=prefer_corrected,
                         prefer_filtered=prefer_filtered
                         )

            drtmd.fit_all(ignore_errors=True, refit=False)
            # drtmd.fit_observation(0)

            if score_badness:
                drtmd.score_group_data_badness(step_id + group_id_suffix, ['eta'])
                drtmd.score_group_fit_badness(step_id + group_id_suffix, ['eta'], include_special=True)

            if resolve:
                if resolve_kw is None:
                    resolve_kw = dict(psi_sort_dims=['eta'], sigma=0.75, lambda_psi=75,
                                      tau_filter_sigma=0.35, special_filter_sigma=0.35)
                print('resolve_kw:', resolve_kw)
                drtmd.resolve_group(step_id + group_id_suffix, **resolve_kw)

        else:
            # Reached end of steps in data_path
            # Save fits
            fname = get_fname()
            drtmd.save_attributes('all', fit_path.joinpath(fname))
            print(f'Saved {fname}')

            break

