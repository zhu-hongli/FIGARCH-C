import numpy as np
from math import gamma
from arch.univariate.volatility import VolatilityProcess, VarianceForecast, initial_value_warning, InitialValueWarning
from warnings import warn


class FIGARCH_C(VolatilityProcess):
    def __init__(self, p=1, q=1, power=2.0, truncation=1000):
        super(FIGARCH_C, self).__init__()
        self.p = int(p)
        self.q = int(q)
        self.power = power
        self.num_params = 2 + p + q
        self._truncation = int(truncation)
        if p < 0 or q < 0 or p > 1 or q > 1:
            raise ValueError('p and q must be either 0 or 1.')
        if self._truncation <= 0:
            raise ValueError('truncation must be a positive integer')
        if power <= 0.0:
            raise ValueError('power must be strictly positive, usually larger than 0.25')
        self.name = self._name()

    @property
    def truncation(self):
        """Truncation lag for the ARCH-infinity approximation"""
        return self._truncation

    def __str__(self):
        descr = self.name

        if self.power != 1.0 and self.power != 2.0:
            descr = descr[:-1] + ', '
        else:
            descr += '('
        for k, v in (('p', self.p), ('q', self.q)):
            descr += k + ': ' + str(v) + ', '
        descr = descr[:-2] + ')'

        return descr

    def variance_bounds(self, resids, power=2.0):
        return super(FIGARCH_C, self).variance_bounds(resids, self.power)

    def _name(self):
        q, power = self.q, self.power
        if power == 2.0:
            if q == 0:
                return 'FIARCH'
            else:
                return 'FIGARCH'
        elif power == 1.0:
            if q == 0:
                return 'FIAVARCH'
            else:
                return 'FIAVGARCH'
        else:
            if q == 0:
                return 'Power FIARCH (power: {0:0.1f})'.format(self.power)
            else:
                return 'Power FIGARCH (power: {0:0.1f})'.format(self.power)

    def bounds(self, resids):
        v = np.mean(abs(resids) ** self.power)

        bounds = [(0.0, 10.0 * v)]
        bounds.extend([(0.0, 0.5)] * self.p)  # phi
        bounds.extend([(0.0, 1.0)])  # d
        bounds.extend([(0.0, 1.0)] * self.q)  # beta

        return bounds

    def constraints(self):

        # omega > 0 <- 1
        # 0 <= d <= 1 <- 2
        # 0 <= phi <= (1 - d) / 2 <- 2
        # 0 <= beta <= d + phi <- 2
        a = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, -2, -1, 0],
                      [0, 0, 1, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1],
                      [0, 1, 1, -1]])
        b = np.array([0, 0, -1, 0, -1, 0, 0])
        if not self.q:
            a = a[:-2, :-1]
            b = b[:-2]
        if not self.p:
            # Drop column 1 and rows 1 and 2
            a = np.delete(a, (1,), axis=1)
            a = np.delete(a, (1, 2), axis=0)
            b = np.delete(b, (1, 2))

        return a, b

    def compute_variance(self, parameters, resids, sigma2, backcast,
                         var_bounds):
        # fresids is abs(resids) ** power
        power = self.power
        fresids = np.abs(resids) ** power

        p, q, truncation = self.p, self.q, self.truncation

        nobs = resids.shape[0]
        figarch_recursion_cap(parameters, fresids, sigma2, p, q, nobs, truncation, backcast,
                          var_bounds)
        inv_power = 2.0 / power
        sigma2 **= inv_power

        return sigma2

    def backcast_transform(self, backcast):
        backcast = super(FIGARCH_C, self).backcast_transform(backcast)
        return np.sqrt(backcast) ** self.power

    def backcast(self, resids):
        power = self.power
        tau = min(75, resids.shape[0])
        w = (0.94 ** np.arange(tau))
        w = w / sum(w)
        backcast = np.sum((abs(resids[:tau]) ** power) * w)

        return backcast

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        truncation = self.truncation
        p, q, power = self.p, self.q, self.power
        lam = figarch_weights_cap(parameters[1:], p, q, truncation)
        lam_rev = lam[::-1]
        errors = rng(truncation + nobs + burn)

        if initial_value is None:
            persistence = np.sum(lam)
            beta = parameters[-1] if q else 0.0

            initial_value = parameters[0]
            if beta < 1:
                initial_value /= (1 - beta)
            if persistence < 1:
                initial_value /= (1 - persistence)
            if persistence >= 1.0 or beta >= 1.0:
                warn(initial_value_warning, InitialValueWarning)

        sigma2 = np.empty(truncation + nobs + burn)
        data = np.empty(truncation + nobs + burn)
        fsigma = np.empty(truncation + nobs + burn)
        fdata = np.empty(truncation + nobs + burn)

        fsigma[:truncation] = initial_value
        sigma2[:truncation] = initial_value ** (2.0 / power)
        data[:truncation] = np.sqrt(sigma2[:truncation]) * errors[:truncation]
        fdata[:truncation] = abs(data[:truncation]) ** power
        omega = parameters[0]
        beta = parameters[-1] if q else 0
        omega_tilde = omega / (1 - beta)
        for t in range(truncation, truncation + nobs + burn):
            fsigma[t] = omega_tilde + lam_rev.dot(fdata[t - truncation:t])
            sigma2[t] = fsigma[t] ** (2.0 / power)
            data[t] = errors[t] * np.sqrt(sigma2[t])
            fdata[t] = abs(data[t]) ** power

        return data[truncation + burn:], sigma2[truncation + burn:]

    def starting_values(self, resids):
        truncation = self.truncation
        ds = [.2, .5, .7]
        phi_ratio = [.2, .5, .8] if self.p else [0]
        beta_ratio = [.1, .5, .9] if self.q else [0]

        power = self.power
        target = np.mean(abs(resids) ** power)
        scale = np.mean(resids ** 2) / (target ** (2.0 / power))
        target *= (scale ** (power / 2))

        svs = []
        for d in ds:
            for pr in phi_ratio:
                phi = (1 - d) / 2 * pr
                for br in beta_ratio:
                    beta = (d + phi) * br
                    temp = [phi, d, beta]
                    lam = figarch_weights_cap(np.array(temp), 1, 1, truncation)
                    omega = (1 - beta) * target * (1 - np.sum(lam))
                    svs.append((omega, phi, d, beta))
        svs = set(svs)
        svs = [list(sv) for sv in svs]
        svs = np.array(svs)
        if not self.q:
            svs = svs[:, :-1]
        if not self.p:
            svs = np.c_[svs[:, [0]], svs[:, 2:]]

        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        llfs = np.zeros(len(svs))
        for i, sv in enumerate(svs):
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)

        return svs[int(loc)]

    def parameter_names(self):
        names = ['omega']
        if self.p:
            names += ['phi']
        names += ['d']
        if self.q:
            names += ['beta']
        return names

    def _check_forecasting_method(self, method, horizon):
        if horizon == 1:
            return

        if method == 'analytic' and self.power != 2.0:
            raise ValueError('Analytic forecasts not available for horizon > 1 when power != 2')
        return

    def _analytic_forecast(self, parameters, resids, backcast, var_bounds, start, horizon):
        sigma2, forecasts = self._one_step_forecast(parameters, resids, backcast,
                                                    var_bounds, horizon)
        if horizon == 1:
            forecasts[:start] = np.nan
            return VarianceForecast(forecasts)

        truncation = self.truncation
        p, q = self.p, self.q
        lam = figarch_weights_cap(parameters[1:], p, q, truncation)
        lam_rev = lam[::-1]
        t = resids.shape[0]
        omega = parameters[0]
        beta = parameters[-1] if q else 0.0
        omega_tilde = omega / (1 - beta)
        temp_forecasts = np.empty(truncation + horizon)
        resids2 = resids ** 2
        for i in range(start, t):
            available = i + 1 - max(0, i - truncation + 1)
            temp_forecasts[truncation - available:truncation] = resids2[
                                                                max(0, i - truncation + 1):i + 1]
            if available < truncation:
                temp_forecasts[:truncation - available] = backcast
            for h in range(horizon):
                lagged_forecasts = temp_forecasts[h:truncation + h]
                temp_forecasts[truncation + h] = omega_tilde + lam_rev.dot(lagged_forecasts)
            forecasts[i, :] = temp_forecasts[truncation:]

        forecasts[:start] = np.nan
        return VarianceForecast(forecasts)

    def _simulation_forecast(self, parameters, resids, backcast, var_bounds, start, horizon,
                             simulations, rng):
        sigma2, forecasts = self._one_step_forecast(parameters, resids, backcast,
                                                    var_bounds, horizon)
        t = resids.shape[0]
        paths = np.full((t, simulations, horizon), np.nan)
        shocks = np.full((t, simulations, horizon), np.nan)

        power = self.power

        truncation = self.truncation
        p, q = self.p, self.q
        lam = figarch_weights_cap(parameters[1:], p, q, truncation)
        lam_rev = lam[::-1]
        t = resids.shape[0]
        omega = parameters[0]
        beta = parameters[-1] if q else 0.0
        omega_tilde = omega / (1 - beta)
        fpath = np.empty((simulations, truncation + horizon))
        fresids = np.abs(resids) ** power

        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            available = i + 1 - max(0, i - truncation + 1)
            fpath[:, truncation - available:truncation] = fresids[max(0, i + 1 - truncation):i + 1]
            if available < truncation:
                fpath[:, :(truncation - available)] = backcast
            for h in range(horizon):
                # 1. Forecast transformed variance
                lagged_forecasts = fpath[:, h:truncation + h]
                temp = omega_tilde + lagged_forecasts.dot(lam_rev)
                # 2. Transform variance
                sigma2 = temp ** (2.0 / power)
                # 3. Simulate new residual
                shocks[i, :, h] = std_shocks[:, h] * np.sqrt(sigma2)
                paths[i, :, h] = sigma2
                forecasts[i, h] = sigma2.mean()
                # 4. Transform new residual
                fpath[:, truncation + h] = np.abs(shocks[i, :, h]) ** power

        forecasts[:start] = np.nan
        return VarianceForecast(forecasts, paths, shocks)

def figarch_recursion_cap(parameters, fresids, sigma2, p, q, nobs, trunc_lag, backcast,
                             var_bounds):
    """
    Parameters
    ----------
    parameters : ndarray
        Model parameters of the form (omega, phi, d, beta) where omega is the
        intercept, d is the fractional integration coefficient and phi and beta
        are parameters of the volatility process.
    fresids : ndarray
        Absolute value of residuals raised to the power in the model.  For
        example, in a standard GARCH model, the power is 2.0.
    sigma2 : ndarray
        Conditional variances with same shape as resids
    p : int
        0 or 1 to indicate whether the model contains phi
    q : int
        0 or 1 to indicate whether the model contains beta
    nobs : int
        Length of resids
    trunc_lag : int
        Truncation lag for the ARCH approximations
    backcast : float
        Value to use when initializing the recursion
    var_bounds : ndarray
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period

    Returns
    -------
    sigma2 : ndarray
        Conditional variances
    """

    omega = parameters[0]
    beta = parameters[1 + p + q] if q else 0.0
    omega_tilde = omega / (1-beta)
    lam = figarch_weights_cap(parameters[1:], p, q, trunc_lag)
    for t in range(nobs):
        bc_weight = 0.0
        for i in range(t, trunc_lag):
            bc_weight += lam[i]
        sigma2[t] = omega_tilde + bc_weight * backcast
        for i in range(min(t, trunc_lag)):
            sigma2[t] += lam[i] * fresids[t - i - 1]
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

    return sigma2

def figarch_weights_cap(parameters, p, q, truncation):
    r"""
    Parameters
    ----------
    parameters : ndarray
        Model parameters of the form (omega, phi, d, beta) where omega is the
        intercept, d is the fractional integration coefficient and phi and beta
        are parameters of the volatility process.
    p : int
        0 or 1 to indicate whether the model contains phi
    q : int
        0 or 1 to indicate whether the model contains beta
    trunc_lag : int
        Truncation lag for the ARCH approximations

    Returns
    -------
    lam : ndarray
        ARCH(:math:`\infty`) coefficients used to approximate model dynamics
    """
    phi = parameters[0] if p else 0.0
    d = parameters[1] if p else parameters[0]
    beta = parameters[p + q] if q else 0.0

    # Recursive weight computation
    lam = np.empty(truncation)
    lam[0]=-beta+phi/gamma(2-d)-2**(1-d)+2+beta*(1-1/gamma(2-d))
    delta = np.empty(truncation)
    for i in range(0, truncation):
        delta[i] = ((i + 2) ** (1 - d) - 2 * (i + 1) ** (1 - d) + (i) ** (1 - d)) / gamma(2 - d)
    for i in range(1, truncation):
        lam[i] = beta * lam[i - 1] + (-delta[i] + phi * delta[i - 1])
        lam[i] *= gamma(2-d)
    lam[0] *= gamma(2-d)
    return lam

def bounds_check(sigma2, var_bounds):
    if sigma2 < var_bounds[0]:
        sigma2 = var_bounds[0]
    elif sigma2 > var_bounds[1]:
        if not np.isinf(sigma2):
            sigma2 = var_bounds[1] + np.log(sigma2 / var_bounds[1])
        else:
            sigma2 = var_bounds[1] + 1000
    return sigma2


if __name__ == "__main__":
    from data_pre_process import get_sp500_log_returns
    from arch.univariate import Normal, ConstantMean
    data = get_sp500_log_returns()
    am = ConstantMean(data)
    am.volatility = FIGARCH_C()
    res = am.fit()
    print(res.summary())