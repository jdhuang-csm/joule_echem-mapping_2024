import os
import pathlib
from pathlib import Path
import numpy as np

import hybdrt.fileload as fl


def get_mode_step(file):
    if type(file) == pathlib.WindowsPath:
        fname = file.name
    else:
        fname = os.path.basename(file)

    # Remove filtered tag
    fname = fname.replace('_Filtered', '')

    # Get step text
    txt = fname.split('_')[-1].replace('.DTA', '')

    # Remove chrono suffix if present
    txt = txt.split('-')[0]

    return int(txt)


def get_pmap_files(data_path, step_id, data_type='hybrid', mode=None,
                   include_full_eis=False, include_ocv=True, prefer_corrected=False, prefer_filtered=False):
    if type(data_path) == str:
        data_path = Path(data_path)

    # Validate data type
    check_data_type(data_type)

    # Select mode
    if mode is None:
        modes = ['discharge', 'charge']
    elif type(mode) == str:
        modes = [mode]
    else:
        modes = mode

    # Determine chrono file suffix
    if data_type in ['hybrid', 'chrono']:
        if len(list(data_path.glob(f'CHRONOP_Step{step_id}*-{modes[0]}*-a.DTA'))) > 0:
            chrono_suffix = '-a'
        else:
            chrono_suffix = ''

        # if prefer_filtered:
        #     chrono_suffix = chrono_suffix + '_Filtered'
    else:
        chrono_suffix = None

    chrono_files = []
    eis_files = []
    for mode in modes:
        chrono_mode_files = None
        eis_mode_files = None

        if data_type in ['hybrid', 'chrono']:
            chrono_pattern = f'CHRONOP_Step{step_id}*-{mode}*{chrono_suffix}.DTA'
            chrono_mode_files = list(data_path.glob(chrono_pattern))

            if prefer_filtered:
                # Find filtered files and replace originals
                filtered_pattern = chrono_pattern.replace('.DTA', '_Filtered.DTA')
                filtered_files = list(data_path.glob(filtered_pattern))
                orig_files = [data_path.joinpath(f.name.replace('_Filtered', '')) for f in filtered_files]
                for of, ff in zip(orig_files, filtered_files):
                    chrono_mode_files[chrono_mode_files.index(of)] = ff

            if prefer_corrected:
                # Find corrected files and replace originals
                corrected_pattern = chrono_pattern.replace('.DTA', '.DTA-CORRECTED')
                corrected_files = list(data_path.glob(corrected_pattern))
                orig_files = [data_path.joinpath(f.name.replace('DTA-CORRECTED', 'DTA')) for f in corrected_files]
                for of, cf in zip(orig_files, corrected_files):
                    chrono_mode_files[chrono_mode_files.index(of)] = cf

            # Sort by step number
            chrono_mode_files = sorted(chrono_mode_files, key=get_mode_step)

        if data_type in ['hybrid', 'eis']:
            eis_mode_files = list(data_path.glob(f'EISGALV_Step{step_id}*-{mode}*.DTA'))
            # Exclude post-staircase EIS files
            eis_mode_files = [f for f in eis_mode_files if f.name.find(f'{mode}_Post') < 0]

            if prefer_corrected:
                # Find corrected files and replace originals
                corrected_files = list(data_path.glob(f'EISGALV_Step{step_id}*-{mode}*.DTA-CORRECTED'))
                orig_files = [data_path.joinpath(f.name.replace('DTA-CORRECTED', 'DTA')) for f in corrected_files]
                for of, cf in zip(orig_files, corrected_files):
                    eis_mode_files[eis_mode_files.index(of)] = cf

            # Sort by step number
            eis_mode_files = sorted(eis_mode_files, key=get_mode_step)

        if chrono_mode_files is None:
            chrono_mode_files = [None] * len(eis_mode_files)

        if eis_mode_files is None:
            eis_mode_files = [None] * len(chrono_mode_files)

        if len(eis_files) != len(chrono_files):
            raise ValueError(f'Mismatched chrono and EIS files. Found {len(chrono_files)} chrono files '
                             f'and {len(eis_files)} EIS files for Step {step_id}')

        chrono_files += chrono_mode_files
        eis_files += eis_mode_files

    # Add full EIS files
    if include_full_eis:
        full_eis_files = []

        if include_ocv:
            full_eis_files += list(data_path.glob(f'EISPOT_Step{step_id}_Measure_Cycle0.DTA'))
        for mode in modes:
            full_eis_files += list(data_path.glob(f'EISGALV_Step{step_id}*Staircase-{mode}_Post.DTA'))

        if prefer_corrected:
            # Replace original files with corrected files
            for i, file in enumerate(full_eis_files):
                cor_file = list(data_path.glob(file.name + '-CORRECTED'))
                if len(cor_file) > 0:
                    full_eis_files[i] = cor_file[0]

        eis_files += full_eis_files
        chrono_files += [None] * len(full_eis_files)

    return chrono_files, eis_files


def get_step_ocv(data_path, step_id):
    if type(data_path) == str:
        data_path = Path(data_path)

    # Load OCV data
    ocv_file = next(data_path.glob(f'OCP_Step{step_id}_Measure_Cycle0.DTA'))
    df = fl.read_curve(ocv_file)

    return df['Vf'].median()


def get_init_timestamp(cell_path):
    cell_path = Path(cell_path)
    reduction_path = next(cell_path.glob('run0*'))
    ocp_file = reduction_path.joinpath('OCP_Step4_Measure_Cycle0.DTA')
    if os.path.exists(ocp_file):
        return fl.get_timestamp(ocp_file)
    else:
        breakin_path = next(cell_path.glob('run1*'))
        ocp_file = breakin_path.joinpath('OCP_Step0a_Measure_Cycle0.DTA')
        return fl.get_timestamp(ocp_file)


def add_pmap_obs(drtmd, data_path, step_ids, psi_fields, init_time=None, i_sign_convention=1,
                 area=None, data_type='hybrid', mode=None, include_full_eis=False, prefer_corrected=False,
                 prefer_filtered=False, group_id_suffix='',
                 fit=False, show_progress=True):
    if type(step_ids) == str:
        step_ids = [step_ids]

    for step_id in step_ids:
        chrono_files, eis_files = get_pmap_files(data_path, step_id, data_type=data_type, mode=mode,
                                                 include_full_eis=include_full_eis,
                                                 prefer_corrected=prefer_corrected,
                                                 prefer_filtered=prefer_filtered
                                                 )
        num_obs = len(chrono_files)
        print_interval = np.ceil(num_obs / 4)

        # Get OCV
        v_oc = get_step_ocv(data_path, step_id)

        # Get timestamp for step (treat all files for step as same time)
        # TODO: consider getting actual time for each measurement within the step
        if init_time is not None:
            if data_type in ('hybrid', 'chrono'):
                step_timestamp = fl.get_timestamp(chrono_files[0])
            else:
                step_timestamp = fl.get_timestamp(eis_files[0])
            step_time_delta = (step_timestamp - init_time).total_seconds() / 60
        else:
            step_time_delta = None

        if show_progress:
            print(f'Adding {num_obs} observations for step {step_id}...')

        for i, (chrono_file, eis_file) in enumerate(zip(chrono_files, eis_files)):
            # Get notes
            if chrono_file is not None:
                notes = fl.read_notes(chrono_file, parse=True)
            else:
                notes = fl.read_notes(eis_file, parse=True)

            # Check EIS data
            valid_eis = True
            if eis_file is not None:
                eis_df = fl.read_eis(eis_file)
                if eis_df['Zmod'].min() > 100:
                    print(f'Found corrupted EIS file: {eis_file.name}. '
                          'The corresponding observation will be excluded')
                    valid_eis = False
                    eis_file = None
            else:
                eis_df = None

            if valid_eis:
                # Get i/v data
                if chrono_file is not None and chrono_file.name.find('Staircase') > 0:
                    mode_step = get_mode_step(chrono_file)
                else:
                    mode_step = 0

                if chrono_file is None:  # or (mode_step > 2 and eis_file is not None)
                    current = eis_df['Idc'].median()
                    voltage = eis_df['Vdc'].median()
                else:
                    # Sometimes see incorrect DC i/v in first or second EIS file
                    chrono_data = fl.get_chrono_tuple(fl.read_chrono(chrono_file))
                    _, i_sig, v_sig = chrono_data
                    current = np.median(i_sig[-20:])
                    voltage = np.median(v_sig[-20:])

                # Normalize current to cell area
                if area is not None:
                    current = current / area

                # Get psi vector
                # Extract info from notes
                try:
                    psi_i = [float(notes[field]) for field in psi_fields]
                except KeyError:
                    for field in psi_fields:
                        if field not in notes.keys():
                            raise KeyError(f'Field {field} not found in notes. Available fields: {list(notes.keys())}')

                # Append current and voltage
                psi_i = psi_i + [current * i_sign_convention, voltage, voltage - v_oc]

                # Add elapsed time
                if step_time_delta is not None:
                    psi_i = psi_i + [step_time_delta]

                drtmd.add_observation(psi_i, chrono_file, eis_file, group_id=step_id + group_id_suffix, fit=fit)

            if show_progress and ((i + 1) % print_interval == 0 or i == num_obs - 1):
                print('{} / {}'.format(i + 1, num_obs))


def check_data_type(data_type):
    data_types = ['hybrid', 'chrono', 'eis']
    if data_type not in data_types:
        raise ValueError(f'Invalid data_type {data_type}. Options: {data_types}')
