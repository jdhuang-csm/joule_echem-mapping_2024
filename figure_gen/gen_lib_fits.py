import os
import numpy as np
from pathlib import Path

import hybdrt.fileload as fl
from hybdrt import mapping

import battmap.fit
from battmap.reader import ReaderCollection


# Set path in which to save fits
script_path = Path(__file__)
fit_path = script_path.parent.joinpath('fits', 'LIB')

# Path in which data is located
datadir = script_path.parent.joinpath('../data/LIB/mapping')

# Rated cell capacity (mAh)
tot_cap = 3400

# Make DRTMD instance and set fit parameters
tau_supergrid = np.logspace(-7, 3, 101)
basis_nu = np.concatenate([np.linspace(-1, -0.4, 25), np.linspace(0.4, 1, 25)])
mrt = mapping.DRTMD(
    tau_supergrid=tau_supergrid, fixed_basis_nu=basis_nu,
)

mrt.fit_dop = True
mrt.fit_ohmic = True
mrt.fit_capacitance = False
mrt.fit_inductance = False

# State variable dimensions
mrt.psi_dim_names = ['c_rate', 'soc', 'time_under_load', 'total_time']


# Instantiate a Reader to offset chrono files for discharge curve
# The Reader is updated when data for each test is loaded
test_reader = ReaderCollection()
mrt.chrono_reader = test_reader


mrt.fit_kw = {
    'nonneg': True,
    'dop_l2_lambda_0': 50,
    'iw_l2_lambda_0': 1e-6,
}

# Weight adjustment to improve resolution of EIS frequency range
wf = np.sqrt(1.5)
mrt.fit_kw['chrono_weight_factor'] = 1 / wf
mrt.fit_kw['eis_weight_factor'] = wf


# Get starting timestamp to track total elapsed time
start_dir = next(datadir.glob('Test1*'))
start_file = next(start_dir.glob('Conditioning_DISCHARGE*.DTA'))
start_timestamp = fl.get_timestamp(start_file)

# Load and fit raw data
for test_id in np.arange(1, 16):
    if test_id <= 5:
        trust_charge_finish = False
    else:
        trust_charge_finish = True
        
    battmap.fit.fit_test_data(mrt, datadir, fit_path, test_id, tot_cap, start_timestamp,
                              suffix='_doplambda=50_NormBySupergrid_wf1.5', filtered=True, seq_downsample_size=1000, 
                              max_num_obs=60,
                              trust_charge_finish=trust_charge_finish, fit_charge=False)