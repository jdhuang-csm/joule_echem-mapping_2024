import numpy as np

from hybdrt.models import DRT
import hybdrt.preprocessing as pp
import hybdrt.fileload as fl


def fit_sequence_data(cp_seq_tup, downsample=True, downsample_size=1000, bkg_sample_interval=None):
    """
    Fit a DRT instance to test sequence data for background/offset preprocessing
    :param cp_seq_tup:
    :param downsample:
    :param downsample_size:
    :return:
    """
    # Unpack data
    times, i_sig, v_sig = cp_seq_tup

    # Keep the first step only
    step_times, _ = pp.get_step_info(times, i_sig, allow_consecutive=False, offset_step_times=True)
    step_times = step_times[:1]

    # Downsample by interval since data is highly non-uniform
    if downsample:
        downsample_interval = int(np.floor(len(times) / downsample_size))
        print('Sequence data downsample interval:', downsample_interval)
        if downsample_interval > 1:
            times = times[::downsample_interval]
            i_sig = i_sig[::downsample_interval]
            v_sig = v_sig[::downsample_interval]
            cp_seq_tup = (times, i_sig, v_sig)

    if bkg_sample_interval is None:
        bkg_sample_interval = (times[-1] - times[0]) / 500

    drt_seq = DRT(fit_dop=False)
    drt_seq.fit_chrono(*cp_seq_tup, max_iter=10, l2_lambda_0=10,
                       step_times=step_times, nonneg=False,
                       # downsample=downsample,
                       # downsample_kw={
                       #     'method': 'match',
                       #     'target_times': target_times
                       # },
                       subtract_background=True, background_type='static',
                       background_corr_power=None,
                       estimate_background_kw={
                           'linear_sample_interval': bkg_sample_interval,
                           'length_scale_bounds': (bkg_sample_interval, bkg_sample_interval * 100),
                           'kernel_size': 1, 'n_restarts': 2,
                           'noise_level_bounds': (1e-4, 1)
                       }
                       )

    return drt_seq


def offset_chrono_data(chrono_data, init_timestamp, drt_seq):
    # Offset time
    chrono_data['elapsed'] = (chrono_data['timestamp'] - init_timestamp).dt.total_seconds()

    # Get baseline voltage
    v_base = drt_seq.predict_response(chrono_data['elapsed'].values, subtract_background=False)

    # Offset voltage
    chrono_data['Vf'] = chrono_data['Vf'] - v_base

    return chrono_data


def make_chrono_reader(init_timestamp, drt_seq):
    """
    Make chrono reader function for DRTMD
    :param init_timestamp:
    :param drt_seq:
    :return:
    """
    def read_func(chrono_file):
        cp_df = fl.read_chrono(chrono_file)
        offset_df = offset_chrono_data(cp_df, init_timestamp, drt_seq)
        return fl.get_chrono_tuple(offset_df)

    return read_func






