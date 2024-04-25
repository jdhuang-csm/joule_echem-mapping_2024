import numpy as np
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from hybdrt.evaluation import kl_div_array
from hybdrt.evaluation import r2_score


def score_rq_gamma(tau, gamma_drt, gamma_rq):
    r2 = r2_score(gamma_drt.flatten(), gamma_rq.flatten())
    # Normalize
    gamma_norms = []
    for g in [gamma_drt, gamma_rq]:
        Rp = np.trapz(g, x=np.log(tau), axis=-1)
        g_norm = g / Rp[..., None]
        gamma_norms.append(g_norm)

    kld_arr = kl_div_array(np.log(tau), *gamma_norms)
    kld = np.trapz(kld_arr, x=np.log(tau), axis=-1)
    fkl = np.exp(-2 * kld)

    return r2, fkl


def beta_transform(beta):
    return np.log(beta / (1 - beta))


def beta_invtransform(beta_trans):
    return np.exp(beta_trans) / (1 + np.exp(beta_trans))


class BetaTransform:
    def transform(self, y):
        return beta_transform(y)

    def inv_transform(self, yt):
        return beta_invtransform(yt)


class CustomTransform:
    def __init__(self, mu, scale, prescale=1.0, log_scale=False):
        self.mu = mu
        self.scale = scale
        self.prescale = prescale
        self.log_scale = log_scale

    def transform(self, y):
        y = y * self.prescale
        if self.log_scale:
            y = np.log(y)
        yt = (y - self.mu) / self.scale
        return yt

    def inv_transform(self, yt):
        y = yt * self.scale + self.mu
        if self.log_scale:
            y = np.exp(y)
        return y / self.prescale


class PressureTransform():
    def __init__(self, pres_indices, log_scale=True):
        self.pres_indices = pres_indices
        self.log_scale = log_scale
        self.column_scales = None

    def fit(self, X, y, **fit_params):
        X = X.copy()
        if self.log_scale:
            X[:, self.pres_indices] = np.log(X[:, self.pres_indices])

        self.column_scales = np.std(X, axis=0)

        # Get pressure std across all gases for consistent scaling
        pres_std = np.std(X[:, self.pres_indices])
        self.column_scales[self.pres_indices] = pres_std
        return self

    def transform(self, X):
        Xt = X.copy()
        return Xt / self.column_scales[None, :]


class ChainTransform:
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def transform(self, y):
        yt = y.copy()
        for transform in self.transform_list:
            yt = transform.transform(yt)

        return yt

    def inv_transform(self, yt):
        y = yt.copy()
        for transform in self.transform_list[::-1]:
            y = transform.inv_transform(y)

        return y


class YTransformRegressor:
    def __init__(self, reg, y_transform):
        self.reg = reg
        self.y_transform = y_transform

    def fit(self, X, y, **fit_params):
        yt = self.y_transform.transform(y)
        self.reg.fit(X, yt, **fit_params)
        return self

    def predict(self, X):
        yt = self.reg.predict(X)
        return self.y_transform.inv_transform(yt)

    def __sklearn_is_fitted__(self):
        check_is_fitted(self.reg)
        return True

    @property
    def _estimator_type(self):
        return "regressor"


def gps_from_stan(stan_data, stan_results, X_train, extract_kernel=True, pres_indices=[0, 1, 2],
                  noise_level=None, which=['R','tau']):
    
    gps = {'R': [], 'tau': [], 'beta': []}
    for key in list(gps.keys()):
        if key not in which:
            del gps[key]

    pres_trans = PressureTransform(pres_indices=pres_indices, log_scale=True)

    for k in range(stan_data['K']):
        if extract_kernel:
            const = kernels.ConstantKernel(stan_results['var']['alpha'][k], constant_value_bounds='fixed')
            rbf = kernels.RBF(length_scale=stan_results['var']['rho'][k], length_scale_bounds='fixed')
            if noise_level is None:
                sq_sigma = (stan_results['var']['sigma'][k] * stan_data['sigma_gp_scale']) ** 2
            else:
                sq_sigma = noise_level
            white = kernels.WhiteKernel(noise_level=sq_sigma,
                                        noise_level_bounds='fixed')
            kernel_k = const * rbf + white
        else:
            # Bound length scale for time dimension to prevent overfitting
            kernel_k = kernels.RBF(length_scale=(1, 1, 1, 1, 10), 
                                   length_scale_bounds=[(0.01, 1000)] * 4 + [(5, 1000)])
            if noise_level is None:
                kernel_k += kernels.WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 100))
            else:
                kernel_k += kernels.WhiteKernel(noise_level=noise_level, noise_level_bounds='fixed')

        R_transform = CustomTransform(mu=stan_results['var']['lnR_mu'][k], scale=stan_results['var']['lnR_scale'][k],
                                      prescale=stan_results['R_scale'], log_scale=True
                                      )

        tau_transform = CustomTransform(mu=stan_results['var']['lntau_mu'][k],
                                        scale=stan_results['var']['lntau_scale'][k],
                                        prescale=1, log_scale=True
                                        )

        beta_trans = CustomTransform(mu=stan_results['var']['beta_trans_mu'][k],
                                     scale=stan_results['var']['beta_trans_scale'][k],
                                     prescale=1, log_scale=False
                                     )
        beta_transform = ChainTransform([BetaTransform(), beta_trans])

        # R
        if 'R'in which:
            gp_R_ = GaussianProcessRegressor(kernel=kernel_k, normalize_y=False)
            gp_R_trans = YTransformRegressor(reg=gp_R_, y_transform=R_transform)
            gp_R = Pipeline([('trans', pres_trans), ('gp', gp_R_trans)])
            gp_R.fit(X_train, stan_results['var']['R'][k] / stan_results['R_scale'])
            gps['R'].append(gp_R)

        # tau
        if 'tau' in which:
            gp_tau_ = GaussianProcessRegressor(kernel=kernel_k, normalize_y=False)
            gp_tau_trans = YTransformRegressor(reg=gp_tau_, y_transform=tau_transform)
            gp_tau = Pipeline([('trans', pres_trans), ('gp', gp_tau_trans)])
            gp_tau.fit(X_train, np.exp(stan_results['var']['lntau'][k]))
            gps['tau'].append(gp_tau)

        # beta
        if 'beta'in which:
            gp_beta_ = GaussianProcessRegressor(kernel=kernel_k, normalize_y=False)
            gp_beta_trans = YTransformRegressor(reg=gp_beta_, y_transform=beta_transform)
            gp_beta = Pipeline([('trans', pres_trans), ('gp', gp_beta_trans)])
            gp_beta.fit(X_train, stan_results['var']['beta'][k])
            gps['beta'].append(gp_beta)
            
    return gps
