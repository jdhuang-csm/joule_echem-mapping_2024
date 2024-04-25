import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy
import glob
from pathlib import Path
from scipy.optimize import least_squares
from scipy.integrate import cumtrapz
from scipy.special import factorial
from scipy import ndimage
from skimage import filters
import time
import seaborn as sn
import pandas as pd
import pickle

from cmdstanpy import CmdStanModel

from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.pipeline import Pipeline

from alepython import ale_plot, ale #_first_order_ale_quant, _second_order_ale_quant

import hybdrt
from hybdrt.models import DRT
from hybdrt.models.sequential import fit_sequence
import hybdrt.fileload as fl
import hybdrt.plotting as hplt
import hybdrt.preprocessing as pp

from hybdrt import mapping

import battmap.fit
import battmap.dataload as dl
from battmap import sequence, capacity, surface
from battmap.reader import ReaderCollection

# Set paths
# datadir = Path('G:\\My Drive\\Jake\\Gamry data\\CFCC_4-2\\Batteries\\Molicel_M35A\\Cell1\\hybrid-discharge')


# Set fit path
script_path = Path(__file__)
fit_path = script_path.parent.joinpath('fits', 'LIB')

# Path in which data is located
datadir = script_path.parent.joinpath('../data/LIB/mapping')


tot_cap = 3400

# Make DRTMD instance
tau_supergrid = np.logspace(-7, 3, 101)
basis_nu = np.concatenate([np.linspace(-1, -0.4, 25), np.linspace(0.4, 1, 25)])
mrt = mapping.DRTMD(
    tau_supergrid=tau_supergrid, fixed_basis_nu=basis_nu,
)

mrt.fit_dop = True
mrt.fit_ohmic = True
mrt.fit_capacitance = False
mrt.fit_inductance = False
mrt.psi_dim_names = ['c_rate', 'soc', 'time_under_load', 'total_time']


# Instantiate a Reader to offset chrono files for discharge curve
# The Reader is updated when data for each test is loaded
test_reader = ReaderCollection()
mrt.chrono_reader = test_reader


# mrt_delta = deepcopy(mrt)
# mrt_rbf = deepcopy(mrt)

# mrt_delta.nu_basis_type = 'delta'
# mrt_delta.fit_kw = {
#     'nonneg': True,
#     'dop_l2_lambda_0': 0,
#     'dop_l1_lambda_0': 10
# }

# mrt_rbf.nu_basis_type = 'gaussian'
mrt.fit_kw = {
    'nonneg': True,
    'dop_l2_lambda_0': 50,
    'iw_l2_lambda_0': 1e-6,
}

# Weight adjustment to improve resolution of EIS frequency range
wf = np.sqrt(1.5)
mrt.fit_kw['chrono_weight_factor'] = 1 / wf
mrt.fit_kw['eis_weight_factor'] = wf

# RBF7: max_num_obs = 60, tau_supergrid = np.logspace(-7, 3, 101), iw_l2_lambda_0=1e-6

# Get starting timestamp to track total elapsed time
start_dir = next(datadir.glob('Test1*'))
start_file = next(start_dir.glob('Conditioning_DISCHARGE*.DTA'))
start_timestamp = fl.get_timestamp(start_file)


for test_id in np.arange(1, 16):
    if test_id <= 5:
        trust_charge_finish = False
    else:
        trust_charge_finish = True
        
    battmap.fit.fit_test_data(mrt, datadir, fit_path, test_id, tot_cap, start_timestamp,
                              suffix='_doplambda=50_NormBySupergrid_wf1.5', filtered=True, seq_downsample_size=1000, 
                              max_num_obs=60,
                              trust_charge_finish=trust_charge_finish, fit_charge=False)