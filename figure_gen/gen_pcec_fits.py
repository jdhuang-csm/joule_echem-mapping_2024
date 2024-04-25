import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import time
from copy import deepcopy
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import itertools
import pickle


from skimage import filters, restoration

import pccmap
import pccmap.dataload as dl
from pccmap import clean
from pccmap import fit as pmfit
import pccmap.plot as pcp
    
import hybdrt
import hybdrt.fileload as fl
from hybdrt.models import DRT
from hybdrt.mapping import DRTMD
import hybdrt.plotting as hplt
from hybdrt.utils import stats
from hybdrt import mapping
from hybdrt import filters as hf
from hybdrt.utils.array import nearest_index

# Cell area for current density calculation
d_cell = 0.25
a_cell = np.pi * (d_cell * 2.54/ 2) ** 2


script_path = Path(__file__)
fit_path = script_path.parent.joinpath('fits', 'PCEC')

# cell_path = Path('G:\\My Drive\\Jake\\Gamry data\\CFCC_4-2\\Pub\\Fine-Fine\\220320_220311-2b')
# wet_path = cell_path.joinpath('run3_comprehensive-wet')
datadir = script_path.parent.joinpath('../data/PCEC/mapping')

print(os.path.exists(datadir))
cell_id = 'FF3'

# Set up multi-dimensional DRT (DRTMD)
# ---------------------------------------
psi_note_fields = {
    'Temperature': 'T',
    'NegatrodeFracH2': 'ph2', 
    'PositrodeFracO2': 'po2', 
    'PositrodePH2O': 'ph2o', 
}

psi_dim_names = list(psi_note_fields.values()) + ['j', 'V', 'eta', 'time']

# Set tau and nu grids for DRT-DOP inversion
tau_supergrid = np.logspace(-8, 2, 101)
# Consider only pseudo-inductive DOP elements 
# since no long-timescale pseudo-capacitance is observed
basis_nu = np.linspace(0.4, 1, 25)

# Make DRTMD instance
mrt = DRTMD(tau_supergrid=tau_supergrid, 
            fit_dop=True, fixed_basis_nu=basis_nu, nu_basis_type='gaussian',
            fit_type='drt', warn=False, psi_dim_names=psi_dim_names)

# Fit an ideal ohmic resistance. 
# Do not fit ideal inductance or capacitance - these will be handled by the DOP instead
mrt.fit_ohmic = True
mrt.fit_inductance = False
mrt.fit_capacitance = False

# Set fit kwargs
mrt.fit_kw = dict(
    nonneg=True, 
    dop_l2_lambda_0=50,
    # Background estimation parameters
    subtract_background=True, background_type='dynamic', background_corr_power=0.5,
    estimate_background_kw={
        'length_scale_bounds': (0.05, 5), 
        'n_restarts': 2, 
        'noise_level_bounds': (0.01, 10)
    },
    # Remove extreme values during preprocessing with a simple outlier test
    remove_extremes=True,  
    # Use outlier-robust error structure for fitting
    remove_outliers=False, outlier_p=0.05,
    # Tuning parameters for weight estimation - minor impact
    iw_l1_lambda_0=1e-6, iw_l2_lambda_0=1e-6,
)

# Get the initial timestamp from the initial OCV file
init_file = datadir.joinpath('OCP_PostReduction.DTA')
init_timestamp = fl.get_timestamp(init_file)


mrt.clear_obs()

# Load and fit data
# --------------------
# Each step represents a different set of conditions
for step_num in range(23, 33):
    step_id = f'{step_num}d'
    fit_file = fit_path.joinpath(f'{cell_id}_wet_Step{step_id}_doplambda=50_NormBySupergrid.pkl')
    
    if not os.path.exists(fit_file):
        mrt.clear_obs()
        start = time.time()
        
        # Fit individual measurements
        pmfit.fit_step_data(mrt, datadir, step_id, init_timestamp, a_cell, prefer_filtered=True)

        # NOTE: skip batch refinement here
        # This is done later after flagging and removing corrupted spectra 
        # (see PCEC_mapping.ipynb)
        # resolve_kw = dict(psi_sort_dims=['eta'], sigma=0.75, lambda_psi=75,
        #                   tau_filter_sigma=0, special_filter_sigma=0.)

        # mrt.resolve_group(step_id, **resolve_kw)

        elapsed = time.time() - start
        print('Fit time: {:.1f} minutes'.format(elapsed / 60))

        # Save to file
        mrt.save_attributes('all', fit_file)