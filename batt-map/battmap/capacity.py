import numpy as np

import hybdrt.fileload as fl


def integrate_pstatic_capacity(pstatic_file):
    """
    Integrate potentiostatic charging data (for voltage finish)
    :param pstatic_file: potentiostatic data file
    :return:
    """
    df = fl.read_curve(pstatic_file)
    i_sig = df['Im']
    times = df['Time']
    return 1000 * np.trapz(i_sig, x=times) / 3600


def calc_capacity_delta(elapsed, i_dc):
    """
    Get charge/discharge capacity in mAh
    :param elapsed: elapsed time in seconds
    :param i_dc: DC current in A
    :return:
    """
    return 1000 * elapsed * i_dc / 3600


def calc_soc_delta(elapsed, i_dc, tot_capacity):
    return calc_capacity_delta(elapsed, i_dc) / tot_capacity


def get_data_soc_delta(data, init_timestamp, i_dc, tot_capacity):
    elapsed = (data['timestamp'] - init_timestamp).dt.total_seconds()
    return calc_soc_delta(elapsed, i_dc, tot_capacity)


def get_hybrid_soc(chrono_df, eis_df, init_timestamp, i_dc, tot_capacity, init_soc):
    # Get midpoint SOC
    chrono_soc = init_soc + get_data_soc_delta(chrono_df, init_timestamp, i_dc, tot_capacity)
    eis_soc = init_soc + get_data_soc_delta(eis_df, init_timestamp, i_dc, tot_capacity)
    all_soc = np.concatenate([chrono_soc, eis_soc])

    return 0.5 * (np.max(all_soc) + np.min(all_soc))
