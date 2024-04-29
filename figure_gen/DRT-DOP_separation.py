"""
This script is intended to provide an intuitive demonstration of how DRT and DOP components can be
separated. The first section uses an ordinary ridge regression model to perform DRT-DOP inversion,
showing how an appropriate choice of regularization and scaling prevents the DRT from fitting 
phasance-type impedance features, yielding appropriate DRT and DOP estimates. This section does
NOT use the hybdrt package to avoid the complexity of the self-tuning hierarchical Bayesian model.
As a result, the estimated distributions are sub-optimal, but demonstrate the concept.

The second section uses the full hierarchical Bayesian model implemented in hybdrt to demonstrate
the same concept.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorsys
import scipy
import scipy.linalg
import scipy.special
import cvxopt
from pathlib import Path

import fig_funcs as ff


# Disable cvxopt printing
cvxopt.solvers.options['show_progress'] = False

# Set plot formatting
# def adjust_lightness(color, amount):
# 	try:
# 		c = mpl.colors.cnames[color]
# 	except:
# 		c = color
# 	c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
# 	# print(c)
# 	return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

	
 
# plt.rcParams['font.size'] = 8
# plt.rcParams['legend.fontsize'] = 8
# plt.rcParams['xtick.labelsize'] = 7
# plt.rcParams['ytick.labelsize'] = 7

# half_width = 112 / 25.4 
# full_width = 172 / 25.4 

# default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# # colors = [(74,111,227), (211,63,106), (239,151,8), (17,198,56)]
# # colors = [np.array(c) / 255 * 0.9 for c in colors]

# dark_colors = [adjust_lightness('#0000a7', 1.25), adjust_lightness('#c1272d', 0.8), '#008176']
# base_colors = [adjust_lightness(c, 1.25) for c in dark_colors]
# light_colors = [adjust_lightness(c, 1.25) for c in base_colors]

# data_kw = {'facecolors': 'none', 'c': 'k'}

full_width, half_width, light_colors, base_colors, dark_colors = ff.set_plot_formatting()

data_kw = dict(facecolors='none', edgecolors=[0.1] * 3)

ff.set_plotdir(Path(__file__).parent.joinpath('./figures'))


# ========================================================================================
# This section sets up functions to perform DRT estimation via ordinary ridge regression
# and visualize the results without using the hybrid-drt package.
# ========================================================================================

# Plotting functions
# -------------------
def plot_nyquist(z, ax=None, label='', plot_func='scatter', set_aspect_ratio=True,
                 tight_layout=True, **kw):
    """
    Generate Nyquist plot.
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 2.75))
    else:
        fig = ax.get_figure()

    if plot_func == 'scatter':
        scatter_defaults = {'s': 10, 'alpha': 0.5}
        scatter_defaults.update(kw)
        ax.scatter(z.real, -z.imag, label=label, **scatter_defaults)
    elif plot_func == 'plot':
        ax.plot(z.real, -z.imag, label=label, **kw)
    else:
        raise ValueError(f'Invalid plot type {plot_func}. Options are scatter, plot')

    
    ax.set_xlabel(fr'$Z^\prime$ ($\Omega$)')
    ax.set_ylabel(fr'$-Z^{{\prime\prime}}$ ($\Omega$)')

    # Apply tight_layout before setting aspect ratio
    if tight_layout:
        fig.tight_layout()
        
    if set_aspect_ratio:
        set_nyquist_aspect(ax)
        
    return ax


def set_nyquist_aspect(ax, set_to_axis=None, data=None, center_coords=None):
    fig = ax.get_figure()

    # get data range
    yrng = ax.get_ylim()[1] - ax.get_ylim()[0]
    xrng = ax.get_xlim()[1] - ax.get_xlim()[0]

    # Center on the given coordinates
    if center_coords is not None:
        x_offset = center_coords[0] - (ax.get_xlim()[0] + 0.5 * xrng)
        y_offset = center_coords[1] - (ax.get_ylim()[0] + 0.5 * yrng)
        ax.set_xlim(*np.array(ax.get_xlim()) + x_offset)
        ax.set_ylim(*np.array(ax.get_ylim()) + y_offset)

    # get axis dimensions
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height

    yscale = yrng / height
    xscale = xrng / width

    if set_to_axis is None:
        if yscale > xscale:
            set_to_axis = 'y'
        else:
            set_to_axis = 'x'
    elif set_to_axis not in ['x', 'y']:
        raise ValueError(f"If provided, set_to_axis must be either 'x' or 'y'. Received {set_to_axis}")

    if set_to_axis == 'y':
        # expand the x axis
        diff = (yscale - xscale) * width
        xmin = max(0, ax.get_xlim()[0] - diff / 2)
        mindelta = ax.get_xlim()[0] - xmin
        xmax = ax.get_xlim()[1] + diff - mindelta

        ax.set_xlim(xmin, xmax)
    else:
        # expand the y axis
        diff = (xscale - yscale) * height
        if data is None:
            data_min = 0
        else:
            data_min = np.min(-data['Zimag'])

        if min(data_min, ax.get_ylim()[0]) >= 0:
            # if -Zimag doesn't go negative, don't go negative on y-axis
            ymin = max(0, ax.get_ylim()[0] - diff / 2)
            mindelta = ax.get_ylim()[0] - ymin
            ymax = ax.get_ylim()[1] + diff - mindelta
        else:
            negrng = abs(ax.get_ylim()[0])
            posrng = abs(ax.get_ylim()[1])
            negoffset = negrng * diff / (negrng + posrng)
            posoffset = posrng * diff / (negrng + posrng)
            ymin = ax.get_ylim()[0] - negoffset
            ymax = ax.get_ylim()[1] + posoffset

        ax.set_ylim(ymin, ymax)
        
def zero_axlim(ax, axes):
    """Shift axis limit to zero"""
    for axis in axes:
        lim = np.array(getattr(ax, f'get_{axis}lim')())
        getattr(ax, f'set_{axis}lim')(lim - lim[0])
        
def add_letters(axes, loc=(-0.25, 1.0), size=9, start_index=0):
    """Add letter labels to axes"""
    axes = np.atleast_1d(axes)
    if type(loc) == tuple:
        loc = [loc] * len(axes.ravel())
    for i, ax in enumerate(axes.ravel()):
        ax.text(*loc[i], '{}'.format(chr(97 + start_index + i)).upper(), transform=ax.transAxes, 
                ha='left', va='top', fontsize=size, fontweight='bold')

# Data simulation
# =====================
# Make frequency grid
freq = np.logspace(6, -1, 71)
omega = freq * 2 * np.pi

# Ohmic resistance
R_ohm = 0.1

# Make RQ element
R_rq = 1  # Resistance
tau_rq = 1e-4  # Time constant
n_rq = 0.8  # Dispersion
z_rq = R_rq / (1 + (1j * omega * tau_rq) ** n_rq)

# Make second RQ element
tau_rq2 = 1e-2
z_rq2 = R_rq / (1 + (1j * omega * tau_rq2) ** n_rq)

# True DRT of RQ element
def rq_drt(tau_eval, r, tau_0, n):
    """Analytical DRT of RQ element"""
    nume = r * np.sin((1 - n) * np.pi)
    deno = 2 * np.pi * (np.cosh(n * (np.log(tau_eval) - np.log(tau_0))) - np.cos((1 - n) * np.pi))
    return nume / deno

# Make Warburg element
z0_war = 1  # magnitude
z_war = z0_war * (1j * omega) ** -0.5

# RQ-Warburg circuit
z_rqw = R_ohm + z_rq + z_war

# RQ-RQ circuit
z_2rq = R_ohm + z_rq + z_rq2

# Add Gaussian noise
rng = np.random.default_rng(1289)
sigma = 2e-3
z_rqw_noise = rng.normal(0, sigma, len(freq)) + 1j * rng.normal(0, sigma, len(freq))
z_rqw_noisy = z_rqw + z_rqw_noise

z_2rq_noise = rng.normal(0, sigma, len(freq)) + 1j * rng.normal(0, sigma, len(freq))
z_2rq_noisy = z_2rq + z_2rq_noise

# Plot exact and noisy impedance
fig, axes = plt.subplots(1, 2, figsize=(full_width, full_width * 0.4))
plot_nyquist(z_rqw, plot_func='plot', label='Exact', ax=axes[0], c='k', alpha=0.7)
plot_nyquist(z_rqw_noisy, label='Experiment', ax=axes[0], **data_kw, alpha=0.3)
axes[0].legend()


plot_nyquist(z_2rq, plot_func='plot', label='Exact', ax=axes[1], c='k', alpha=0.7)
plot_nyquist(z_2rq_noisy, label='Experiment', ax=axes[1], **data_kw, alpha=0.3)
axes[1].legend()

fig.tight_layout()

add_letters(axes, loc=(-0.15, 1.))

# Set bottom left corner at origin
for ax in axes:
    zero_axlim(ax, 'xy')
    
    
ff.savefig(fig, 'Supp_drt-dop_separation_spectra')
# plt.show()

# Functions for matrix construction
# ====================================
def gaussian_rbf(y, epsilon):
    return np.exp(-(epsilon * y) ** 2)
        
# DRT impedance matrix
# ---------------------
def drt_integrand(part='real'):
    if part == 'real':
        def func(y, w_n, t_m, epsilon):
            return gaussian_rbf(y, epsilon) / (1 + np.exp(2 * (y + np.log(w_n * t_m))))
    elif part == 'imag':
        def func(y, w_n, t_m, epsilon):
            return -gaussian_rbf(y, epsilon) * np.exp(y) * w_n * t_m / (1 + np.exp(2 * (y + np.log(w_n * t_m))))
    return func

def get_A_drt(omega, tau, epsilon, part):
    integrand_func = drt_integrand(part)
    # A_drt is a Toeplitz matrix - only need to calculate 1st row and column
    w_0 = omega[0]
    t_0 = tau[0]

    y = np.linspace(-20, 20, 1000)
    c = [np.trapz(integrand_func(y, w_n, t_0, epsilon), x=y) for w_n in omega]
    r = [np.trapz(integrand_func(y, w_0, t_m, epsilon), x=y) for t_m in tau]

    A_drt = scipy.linalg.toeplitz(c, r)
    return A_drt

# DOP impedance matrix
# -----------------------
def unit_phasor_impedance(omega, nu):
    return (1j * omega) ** nu

def dop_z_integral_func(nu, omega, nu_m, epsilon):
    # Indefinite integral of DOP basis function times impedance kernel function (Eq. S58)
    out = 0.5 * np.sqrt(np.pi) * unit_phasor_impedance(omega, nu_m) / epsilon
    out *= (1j * omega) ** (np.log(1j * omega) / (4 * epsilon ** 2))
    out *= scipy.special.erf(epsilon * (nu - nu_m) - np.log(1j * omega) / (2 * epsilon))
    return out

def get_nu_limits(nu_m):
    # Get integration limits for DOP basis functions
    a = np.minimum(0, np.sign(nu_m))
    b = np.maximum(0, np.sign(nu_m))
    return a, b

def dop_z_func(omega, nu_m, epsilon):
    # Evaluate integral over finite integration limits (Eq. S57-S58)
    f_int = dop_z_integral_func
    # Get integration limits
    a, b = get_nu_limits(nu_m)
    return f_int(b, omega, nu_m, epsilon) - f_int(a, omega, nu_m, epsilon)

    return func

def get_A_dop(omega, basis_nu, nu_epsilon, normalize=False, tau_c=None):
    nn, ww = np.meshgrid(basis_nu, omega)
    return dop_z_func(ww, nn, nu_epsilon)


# L2 regularization matrices
# Using the integrated penalty formulation derived by Wan et al. in 
# doi:10.1016/J.ELECTACTA.2015.09.097
# ------------------------------
def get_integrated_derivative_func(order=1):
    """
    Create function for integrated derivative matrix

    Parameters:
    -----------
    order : int, optional (default: 1)
        Order of DRT derivative for ridge penalty
    """
    if order == 0:
        def func(x_n, x_m, epsilon):
            a = epsilon * (x_m - x_n)  # epsilon * np.log(l_m / l_n)
            return (np.pi / 2) ** 0.5 * epsilon ** (-1) * np.exp(-(a ** 2 / 2))  # * (epsilon / np.sqrt(np.pi))
    elif order == 1:
        def func(x_n, x_m, epsilon):
            a = epsilon * (x_m - x_n)  # epsilon * np.log(l_m / l_n)
            return -(np.pi / 2) ** 0.5 * epsilon * (-1 + a ** 2) * np.exp(
                -(a ** 2 / 2))  # * (epsilon / np.sqrt(np.pi))
    elif order == 2:
        def func(x_n, x_m, epsilon):
            a = epsilon * (x_m - x_n)
            return (np.pi / 2) ** 0.5 * epsilon ** 3 * (3 - 6 * a ** 2 + a ** 4) * np.exp(-(a ** 2 / 2)) \
                # * (epsilon / np.sqrt(np.pi))
    elif order == 3:
        def func(x_n, x_m, epsilon):
            a = epsilon * (x_m - x_n)
            return -(np.pi / 2) ** 0.5 * epsilon ** 5 * (-15 + (45 * a ** 2) - (15 * a ** 4) + (a ** 6)) \
                    * np.exp(-(a ** 2 / 2))
    else:
        raise ValueError(f'Invalid order {order}. Order must be between 0 and 2')

    return func

def get_L2_matrix(basis_grid, order, epsilon):
    """
    Construct matrix for calculation of ridge regression penalty.
    x.T @ M @ x gives the integral of the squared derivative 
    (of specified order) of the distribtion 
    
    Parameters:
    -----------
    basis_grid : array
        Locations of basis functions
    order : int, optional (default: 1)
        Order of derivative to penalize
    epsilon : float, optional (default: 1)
        Inverse length scale parameter for chosen basis function
    """
    func = get_integrated_derivative_func(order)
    # Matrix is symmetric Toeplitz if basis_type==gaussian and basis_eig is log-uniform.
    # Only need to calculate 1st column
    x_0 = basis_grid[0]

    c = [func(x_n, x_0, epsilon) for x_n in basis_grid]
    
    M = scipy.linalg.toeplitz(c)
    return M

def prep_matrices(reg_order, rescale_drt=True, rescale_dop=True, extend_decades=1):
    """Prepare all matrice for inversion"""
    # NOTE: any changes to this the matrix rescaling here will affect the inversion results 
    # (see notes below);
    # different regularization factors may then be necessary to reproduce the results.
    
    # Configure basis time constants extending 1 decade beyond frequency limits
    logtau_start = -np.log10(omega[0]) - extend_decades
    logtau_end = -np.log10(omega[-1]) + extend_decades
    n_decades = int(logtau_end - logtau_start)
    basis_tau = np.logspace(logtau_start, logtau_end + 1e-8, n_decades * 10 + 1)

    # Construct DRT impedance matrix
    # ------------------------------
    # DRT basis function inverse length scale
    tau_epsilon = 1 / np.diff(np.log(basis_tau))[0]
    tau_epsilon

    # Get the real and imaginary parts of the DRT impedance matrix
    A_drt_re = get_A_drt(omega, basis_tau, tau_epsilon, 'real')
    A_drt_im = get_A_drt(omega, basis_tau, tau_epsilon, 'imag')
    A_drt = A_drt_re + 1j * A_drt_im

    # Rescale the DRT matrix
    # ----------------------
    # NOTE: rescaling the DRT and DOP impedance matrices is an important step to
    # ensure that consistent penalties are applied to both distributions!
    # In the case of conventional DRT estimation, rescaling is unimportant because all
    # DRT basis functions have the same magnitude of influence on the model impedance.
    # However, the DRT and DOP basis functions can have very different magnitudes of influence
    # due to the exponential relationship between the impedance and the DOP.
    if rescale_drt:
        # Normalize all columns to unit maximum to ensure consistent scaling
        drt_scale_factor = np.max(np.abs(A_drt))
    else:
        drt_scale_factor = 1.0
    A_drt = A_drt / drt_scale_factor

    # Construct DOP impedance matrix
    dnu = 0.025
    # Construct the nu grid for DOP basis functions
    # NOTE: As explained in Section S.3.1, we exclude nu values between -0.4 and 0.4 
    # (with the exception of nu=0, i.e. Ohmic resistance) from the nu basis grid.
    # Nu values in this range are rarely observed in practice and are more difficult to separate
    # from Ohmic resistance and RC-type relaxations.
    basis_nu = np.arange(-0.4, -1.01, -dnu)
    nu_epsilon = 1 / dnu
        
    # Rescale the DOP matrix
    # ----------------------
    A_dop_raw = get_A_dop(omega, basis_nu, nu_epsilon, normalize=False)
    if rescale_dop:
        # Normalize all columns to unit median to ensure consistent scaling
        # NOTE: because the DOP impedance has an exponential dependence on frequency,
        # here we normalize by the geometric mean of the modulus, 
        # rather than the maximum, to prevent extreme normalizations.
        # The scaling used in the full model is slightly more complex,
        # as described in the SI, but follows the same logic.
        dop_scale_vector = np.exp(np.mean(np.log(np.abs(A_dop_raw)), axis=0))
    else:
        dop_scale_vector = np.ones(len(basis_nu))
    A_dop_rescaled = A_dop_raw / dop_scale_vector[None, :]

    # Add a column for ohmic resistance (corresponds to delta function at nu=0)
    A_dop = np.empty((A_dop_raw.shape[0], A_dop_raw.shape[1] + 1), dtype=complex)
    A_dop[:, 1:] = A_dop_rescaled #* 10
    A_dop[:, 0] = 1  # Ohmic

    # Construct L2 regularization matrices
    # ------------------------------------
    M_drt = get_L2_matrix(np.log(basis_tau), order=reg_order, epsilon=tau_epsilon)
    M_dop_ = get_L2_matrix(basis_nu, order=reg_order, epsilon=nu_epsilon)
    
    # NOTE: normalizing the L2 penalty matrices is also essential to apply consistent regularization
    # to both distributions! This normalization is equivalent to a compression/expansion of the 
    # DRT and DOP basis grids to obtain equal basis grid spacings.
    # Normalize for consistent penalty magnitudes
    M_drt = M_drt / np.max(M_drt)
    M_dop_ = M_dop_ / np.max(M_dop_)

    # Add regularization row/column for ohmic resistance
    M_dop = np.zeros((M_dop_.shape[0] + 1, M_dop_.shape[1] + 1))
    M_dop[1:, 1:] = M_dop_
    M_dop[0, 0] = 0.1  # Ohmic resistance penalty term
    
    return (
        A_drt, A_dop, M_drt, M_dop, drt_scale_factor, dop_scale_vector, 
        basis_tau, basis_nu, tau_epsilon, nu_epsilon
    )

def concat_matrices(A_drt, A_dop, M_drt, M_dop, lambda_drt, lambda_dop):
    # Concatenate A and M matrices for DRT and DOP
    A_full = np.concatenate((A_drt, A_dop), axis=1)
    M_full = np.zeros((M_drt.shape[1] + M_dop.shape[1], M_drt.shape[1] + M_dop.shape[1]))
    M_full[:M_drt.shape[0], :M_drt.shape[0]] = M_drt * lambda_drt
    
    # Leave the ohmic penalty fixed
    M_dop_scaled = M_dop.copy()
    M_dop_scaled[1:, 1:] = M_dop_scaled[1:, 1:] * lambda_dop
    M_full[M_drt.shape[0]:, M_drt.shape[0]:] = M_dop_scaled
    
    return A_full, M_full

# Solution by convex optimization
# ===============================
def solve_convex_opt(z, A, l2_matrix, l1v, nonneg):
    """
    Solve convex optimization problem
    :param ndarray z: impedance vector
    :param ndarray A: impedance matrix
    :param ndarray l2_matrix: penalty matrix
    :param ndarray l1v: L1 (LASSO) penalty vector
    :param bool nonneg: if True, constrain x >= 0. If False, allow negative values
    """

    p_matrix = (A.T @ A + l2_matrix)
    q_vector = (-A.T @ z + l1v)

    G = -np.eye(p_matrix.shape[1])
    if nonneg:
        h = np.zeros(p_matrix.shape[1])
    else:
        # coefficients can be positive or negative
        h = 10 * np.ones(p_matrix.shape[1])


    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    p_matrix = cvxopt.matrix(p_matrix.T)
    q_vector = cvxopt.matrix(q_vector.T)

    return cvxopt.solvers.qp(p_matrix, q_vector, G, h)

def solve_drtdop(f, z, reg_order, lambda_drt, lambda_dop, 
                 rescale_drt=True, rescale_dop=True, extend_decades=1):
    # Get all matrices
    matrices = prep_matrices(reg_order, rescale_drt, rescale_dop, extend_decades)
    (
        A_drt, A_dop, M_drt, M_dop, drt_scale_factor, dop_scale_vector, _, _, _, _
    ) = matrices
    A_full, M_full = concat_matrices(A_drt, A_dop, M_drt, M_dop, lambda_drt, lambda_dop)
    
    # Convert complex vectors/matrices to concatenated real-valued vectors/matrices
    z_concat = np.concatenate((z.real, z.imag))
    A_concat = np.concatenate((A_full.real, A_full.imag), axis=0)
    
    # Perform convex optimization to obtain parameter vector
    result = solve_convex_opt(z_concat, A_concat, M_full, 0, nonneg=True)
    x = np.array(list(result['x']))
    
    x_drt = x[:A_drt.shape[1]]
    x_dop = x[A_drt.shape[1]:]
    
    return x_drt, x_dop, matrices


# Evaluation and visualization
# ============================
def evaluate_solution(x_drt, x_dop, matrices):
    A_drt = matrices[0]
    A_dop = matrices[1]
    z_drt = A_drt @ x_drt
    z_dop = A_dop[:, 1:] @ x_dop[1:]
    z_ohm = x_dop[0]
    return z_drt, z_dop, z_ohm

def evaluate_distribution(eval_grid, x, basis_grid, epsilon, scale_vector):
    bb, ee = np.meshgrid(basis_grid, eval_grid)
    A = gaussian_rbf(ee - bb, epsilon)
    return A @ (x / scale_vector)


def plot_solution(x_drt, x_dop, matrices, num_rq, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(full_width, full_width * 0.3))
        
    # Unpack matrices
    (
        A_drt, A_dop, M_drt, M_dop, drt_scale_factor, dop_scale_vector, basis_tau, basis_nu,
        tau_epsilon, nu_epsilon
    ) = matrices

    # Plot DRT
    tau_plot = np.logspace(-9, 2, 200)
    
    # Evaluate the true DRT
    gamma_true = rq_drt(tau_plot, R_rq, tau_rq, n_rq)
    if num_rq == 2:
        gamma_true += rq_drt(tau_plot, R_rq, tau_rq2, n_rq)
    
    gamma = evaluate_distribution(np.log(tau_plot), x_drt, np.log(basis_tau), tau_epsilon,
                                  drt_scale_factor)
    axes[0].plot(tau_plot, gamma_true, label='True', c='k', alpha=0.8, ls='--')
    axes[0].plot(tau_plot, gamma, label='Est.', c=light_colors[0], alpha=0.9)
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'$\tau$ (s)')
    axes[0].set_ylabel(r'$\gamma$ ($\Omega$)')

    # Plot DOP
    nu_plot = np.linspace(0, -1, 100)
    rho = evaluate_distribution(nu_plot, x_dop[1:], basis_nu, nu_epsilon, dop_scale_vector)
    if num_rq == 1:
        # Show true DOP of Warburg element (delta function)
        axes[1].axvline(45, c='k', ls=':', label='True')
    axes[1].plot(-nu_plot * 90, rho, label='Est.', c=light_colors[1])
    axes[1].set_xlabel(r'$-\theta$ ($^\circ$)')
    axes[1].set_ylabel(r'$\rho$ ($\Omega \cdot \mathrm{s}^\nu$)')
    # If DOP is very small, keep y-axis at unit range
    if axes[1].get_ylim()[1] < 1:
        axes[1].set_ylim(0, 1)

    # Plot impedance contributions
    z_drt, z_dop, z_ohm = evaluate_solution(x_drt, x_dop, matrices)
    # Measured impedance
    if num_rq == 1:
        z_meas = z_rqw_noisy
    else:
        z_meas = z_2rq_noisy
    plot_nyquist(z_meas, ax=axes[2], label='Exp.', c='k', alpha=0.4)
    # hplt.plot_nyquist((freq, R_ohm + z_rq), ax=axes[2], scale_prefix='', label='RQ exp.')
    plot_nyquist(z_ohm + z_drt, ax=axes[2], plot_func='plot', label='DRT', c=light_colors[0],
                 alpha=0.9)
    # hplt.plot_nyquist((freq, R_rq + R_ohm + z_war), ax=axes[2], scale_prefix='', label='Warburg exp.')
    plot_nyquist(R_rq + R_ohm + z_dop, ax=axes[2], plot_func='plot', label='DOP', 
                 c=light_colors[1], alpha=0.9)
    
    return axes

# ===========================================================================================
# With all functions defined, we can perform some example inversions and examine the results.
# ===========================================================================================
"""
See Section S4.1 of the supplemental text and DRT_nullspace.ipynb for an accompanying discussion. 
In brief, ambiguity exists in the DRT-DOP inversion because the DRT and the DOP are both capable
of reproducing constant-phase impedance. This can result in non-meaningful DRT-DOP inversions
in which a constant-phase feature, such as a Warburg element, is fitted with a combination of 
DRT and DOP peaks. However, the DOP is not capable of reproducing RC-type impedance. Therefore, 
the ambiguity can be resolved by penalizing the DRT more strongly than the DOP, such that the DOP
will always describe any constant-phase features that exist, while the DRT is suppressed in the 
corresponding timescale regions. This is especially visible when we use 2nd-order regularization
in the examples shown below: when the DRT and DOP regularization penalties are equal, the DRT
partially fits the low-frequency Warburg element, resulting in an inaccurate DOP phase estimate.
However, when the DRT penalty is greater than the DOP penalty, the DRT psuedo-peak at long
timescales is suppressed and the DOP estimate moves to the correct phase angle.
"""

# Figure generation
# ===================
for i, (z_noisy, circuit) in enumerate(zip([z_rqw_noisy, z_2rq_noisy], ['RQ-Warburg', 'RQ-RQ'])):
    num_rq = i + 1
    # Order 0 regularization, same regularization strength for DRT and DOP
    # Since each regularization order behaves slightly differently, 
    # a different regularization strength is applied to each order
    fig, axes = plt.subplots(3, 3, figsize=(full_width, full_width * 0.7))
    for i, lambda_0 in enumerate([1e-2, 0.1, 1]):
        x_drt, x_dop, mat = solve_drtdop(freq, z_noisy, reg_order=0,
                                    lambda_drt=lambda_0, lambda_dop=lambda_0
                                    )

        # fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        plot_solution(x_drt, x_dop, mat, num_rq, axes=axes[i])
        axes[i, 0].text(0.05, 0.95, r'$\lambda_{{DRT}}={:.3g}$'.format(lambda_0), 
                        transform=axes[i, 0].transAxes, ha='left', va='top')
        axes[i, 1].text(0.05, 0.95, r'$\lambda_{{DOP}}={:.3g}$'.format(lambda_0),
                        transform=axes[i, 1].transAxes, ha='left', va='top')
        
    axes[0, 0].legend(loc='upper right')
    axes[0, 1].legend(loc='upper right')
    axes[0, 2].legend(loc='upper left')

    # ff.savefig(fig, f'{circuit}_d0_equal')

    # Order 1 regularization, same regularization strength for DRT and DOP
    fig, axes = plt.subplots(3, 3, figsize=(full_width, full_width * 0.7))
    for i, lambda_0 in enumerate([1e-3, 0.1, 10]):
        x_drt, x_dop, mat = solve_drtdop(freq, z_noisy, reg_order=1,
                                    lambda_drt=lambda_0, lambda_dop=lambda_0
                                    )

        # fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        plot_solution(x_drt, x_dop, mat, num_rq, axes=axes[i])
        axes[i, 0].text(0.05, 0.95, r'$\lambda_{{DRT}}={:.3g}$'.format(lambda_0), 
                        transform=axes[i, 0].transAxes, ha='left', va='top')
        axes[i, 1].text(0.05, 0.95, r'$\lambda_{{DOP}}={:.3g}$'.format(lambda_0),
                        transform=axes[i, 1].transAxes, ha='left', va='top')
        
    axes[0, 0].legend(loc='upper right')
    axes[0, 1].legend(loc='upper right')
    axes[0, 2].legend(loc='upper left')

    # ff.savefig(fig, f'{circuit}_d1_equal')

    # Order 2 regularization, same regularization strength for DRT and DOP
    fig, axes = plt.subplots(3, 3, figsize=(full_width, full_width * 0.7))
    for i, lambda_0 in enumerate([0.1, 3, 100]):
        x_drt, x_dop, mat = solve_drtdop(freq, z_noisy, reg_order=2,
                                    lambda_drt=lambda_0, lambda_dop=lambda_0
                                    )

        # fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        plot_solution(x_drt, x_dop, mat, num_rq, axes=axes[i])
        axes[i, 0].text(0.05, 0.95, r'$\lambda_{{DRT}}={:.3g}$'.format(lambda_0), 
                        transform=axes[i, 0].transAxes, ha='left', va='top')
        axes[i, 1].text(0.05, 0.95, r'$\lambda_{{DOP}}={:.3g}$'.format(lambda_0),
                        transform=axes[i, 1].transAxes, ha='left', va='top')
        
    axes[0, 0].legend(loc='upper right')
    axes[0, 1].legend(loc='upper right')
    axes[0, 2].legend(loc='upper left')

    # ff.savefig(fig, f'{circuit}_d2_equal')

    # Order 0 regularization, different regularization strengths for DRT and DOP
    fig, axes = plt.subplots(3, 3, figsize=(full_width, full_width * 0.7))
    lambda_0 = 0.1
    for i, factor in enumerate([0.1, 1, 10]):
        lambda_drt = lambda_0 * factor
        lambda_dop = lambda_0 / factor
        x_drt, x_dop, mat = solve_drtdop(freq, z_noisy, reg_order=0,
                                    lambda_drt=lambda_drt, lambda_dop=lambda_dop
                                    )

        # fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        plot_solution(x_drt, x_dop, mat, num_rq, axes=axes[i])
        axes[i, 0].text(0.05, 0.95, r'$\lambda_{{DRT}}={:.3g}$'.format(lambda_drt), 
                        transform=axes[i, 0].transAxes, ha='left', va='top')
        axes[i, 1].text(0.05, 0.95, r'$\lambda_{{DOP}}={:.3g}$'.format(lambda_dop),
                        transform=axes[i, 1].transAxes, ha='left', va='top')
        
    axes[0, 0].legend(loc='upper right')
    axes[0, 1].legend(loc='upper right')
    axes[0, 2].legend(loc='upper left')

    # ff.savefig(fig, f'{circuit}_d0_ratio')


    # Order 1 regularization, different regularization strengths for DRT and DOP
    fig, axes = plt.subplots(3, 3, figsize=(full_width, full_width * 0.7))
    lambda_0 = 0.5
    for i, factor in enumerate([0.1, 1, 10]):
        lambda_drt = lambda_0 * factor
        lambda_dop = lambda_0 / factor
        x_drt, x_dop, mat = solve_drtdop(freq, z_noisy, reg_order=1,
                                    lambda_drt=lambda_drt, lambda_dop=lambda_dop
                                    )

        # fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        plot_solution(x_drt, x_dop, mat, num_rq, axes=axes[i])
        axes[i, 0].text(0.05, 0.95, r'$\lambda_{{DRT}}={:.3g}$'.format(lambda_drt), 
                        transform=axes[i, 0].transAxes, ha='left', va='top')
        axes[i, 1].text(0.05, 0.95, r'$\lambda_{{DOP}}={:.3g}$'.format(lambda_dop),
                        transform=axes[i, 1].transAxes, ha='left', va='top')
        
    axes[0, 0].legend(loc='upper right')
    axes[0, 1].legend(loc='upper right')
    axes[0, 2].legend(loc='upper left')

    # ff.savefig(fig, f'{circuit}_d1_ratio')

    # Order 2 regularization, different regularization strengths for DRT and DOP
    fig, axes = plt.subplots(3, 3, figsize=(full_width, full_width * 0.7))
    lambda_0 = 1
    for i, factor in enumerate([0.1, 1, 10]):
        lambda_drt = lambda_0 * factor
        lambda_dop = lambda_0 / factor
        x_drt, x_dop, mat = solve_drtdop(freq, z_noisy, reg_order=2,
                                    lambda_drt=lambda_drt, lambda_dop=lambda_dop
                                    )

        # fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        plot_solution(x_drt, x_dop, mat, num_rq, axes=axes[i])
        axes[i, 0].text(0.05, 0.95, r'$\lambda_{{DRT}}={:.3g}$'.format(lambda_drt), 
                        transform=axes[i, 0].transAxes, ha='left', va='top')
        axes[i, 1].text(0.05, 0.95, r'$\lambda_{{DOP}}={:.3g}$'.format(lambda_dop),
                        transform=axes[i, 1].transAxes, ha='left', va='top')
        
    axes[0, 0].legend(loc='upper right')
    axes[0, 1].legend(loc='upper right')
    axes[0, 2].legend(loc='upper left')

    # ff.savefig(fig, f'{circuit}_d2_ratio')

# ===========================================================================================
# This section applies the full hierarchical Bayesian model in hybrid-drt to the same examples
# to show that the same principle applies, but with much improved regularization behavior.
# ===========================================================================================
from hybdrt.models import DRT
import hybdrt.plotting as hplt


drt = DRT(fit_dop=True)


for i, (z_noisy, circuit) in enumerate(zip([z_rqw_noisy, z_2rq_noisy], ['RQ-Warburg', 'RQ-RQ'])):
    num_rq = i + 1

    tau_plot = np.logspace(-9, 2, 200)
    # Evaluate the true DRT
    gamma_true = rq_drt(tau_plot, R_rq, tau_rq, n_rq)
    if num_rq == 2:
        gamma_true += rq_drt(tau_plot, R_rq, tau_rq2, n_rq)

    fig, axes = plt.subplots(3, 3, figsize=(9, 7))

    for i, lambda_dop in enumerate([100, 10, 1]):
        # Fit data with custom DOP regularization strength
        drt.fit_eis(freq, z_noisy, dop_l2_lambda_0=lambda_dop)
        
        # Plot the true DRT
        axes[i, 0].plot(tau_plot, gamma_true, ls='-', c='k', alpha=0.8, label='True')
        # Plot the estimated DRT
        drt.plot_distribution(ax=axes[i, 0], scale_prefix='', label='Est.', c=light_colors[0],
                              alpha=0.9, ls='-.', plot_ci=True)
        
        axes[i, 0].set_xlim(1e-8, 1e2)
        axes[i, 0].set_ylim(0, 0.55)
        
        # Plot the estimated DOP
        drt.plot_dop(ax=axes[i, 1], normalize=True, normalize_tau=(1, 1), scale_prefix='',
                    label='Est.', c=light_colors[1], ls='-.', plot_ci=True, zorder=10)
        
        axes[i, 1].set_xlim(-1, 90)
        axes[i, 1].set_xticks(np.arange(0, 91, 15))
        axes[i, 1].set_ylim(0, 1)
        # Plot the ohmic peak
        axes[i, 1].plot([0, 0], [0, 0.1], c='k', label='True', lw=1)
        # Plot the zero line everywhere else
        axes[i, 1].plot([-1, 90], [0, 0], c='k', lw=1)
        if num_rq == 1:
            # Indicate true Warburg peaks
            axes[i, 1].axvline(45, ls='-', c='k', lw=1)
        
        # Plot DRT and DOP impedance contributions
        hplt.plot_nyquist((freq, z_noisy), ax=axes[i, 2], label='Exp.', scale_prefix='', 
                        **data_kw, alpha=0.4)
        z_drt = drt.predict_z(freq, include_dop=False)
        z_dop = drt.predict_z(freq, include_drt=False)
        z_tot_pred = drt.predict_z(freq)
        hplt.plot_nyquist((freq, z_drt), ax=axes[i, 2], label='DRT', scale_prefix='', 
                        plot_func='plot', c=light_colors[0], alpha=0.9)
        hplt.plot_nyquist((freq, z_dop + R_rq), ax=axes[i, 2], label='DOP', scale_prefix='',
                        plot_func='plot', c=light_colors[1], alpha=0.9)
        # hplt.plot_nyquist((freq, z_tot_pred), ax=axes[i, 2], label='DRT-DOP', scale_prefix='',
        #                 plot_func='plot', c=light_colors[2], alpha=0.9)
        
        # Set bottom left corner to origin
        zero_axlim(axes[i, 2], 'xy')
        
        # Indicate regularization strength
        axes[i, 0].text(0.05, 0.95, '$\lambda_{{DRT}}={:.0f}$'.format(100),
                        transform=axes[i, 0].transAxes, ha='left', va='top')
        axes[i, 1].text(0.05, 0.95, '$\lambda_{{DOP}}={:.0f}$'.format(lambda_dop),
                        transform=axes[i, 1].transAxes, ha='left', va='top')
                        
    axes[0, 0].legend(loc='upper right')
    axes[0, 1].legend(loc='upper right')
    
    add_letters(axes[:, 0], loc=(-0.2, 1.))

    ff.savefig(fig, f'Supp_{circuit}_Full-model')
