import numpy as np
import os
import math
import scipy as sp
import pandas as pd
import pycircstat as pcs
import pyfftw
import numba

# These are the core functions used to identify both spatial and non-spatial phase precession

def circ_lin_corr(circ, lin, ci=None):
    '''1-D Circular-Linear correlation'''

    # Get rid of all the nans in this data 
    nan_index = np.logical_or(np.isnan(circ), np.isnan(lin))
    circ = circ[~nan_index]
    lin = lin[~nan_index]

    if np.size(lin) == 0:
        return np.nan, np.nan, np.nan, np.nan

    def myfun1(p):
        return -np.sqrt(
            (np.sum(np.cos(circ - (p * lin))) / len(circ)) ** 2 + (np.sum(np.sin(circ - (p * lin))) / len(circ)) ** 2)

    # finding the optimal slope, note that we have to restrict the range ofslopes 

    sl = sp.optimize.fminbound(myfun1, (-3 * np.pi) / (np.max(lin) - np.min(lin)), (3 * np.pi) / (
            np.max(lin) - np.min(lin)))  

    # calculate offset
    offs = np.arctan2(np.sum(np.sin(circ - (sl * lin))), np.sum(np.cos(circ - (sl * lin))))  

    # circular-linear correlation:
    linear_circ = np.mod(abs(sl) * lin, 2 * np.pi)  # circular variable derived from the linearization

    # NOTE: ADD FOLLOWING TO THE pycircstat.tests.rayleigh IN ORDER TO COMPUTE A PVAL:
    #    # significance of this correlation coefficient can be tested using the fact that Z is approx. normal
    # l20 = np.mean(np.sin(alpha1 - alpha1_bar)**2)
    # l02 = np.mean(np.sin(alpha2 - alpha2_bar)**2)
    # l22 = np.mean((np.sin(alpha1 - alpha1_bar)**2) * (np.sin(alpha2 - alpha2_bar)**2))
    # z = np.sqrt((len(alpha1) * l20 *l02)/l22) * rho
    # pval = 2 * (1 - stats.norm.cdf(np.abs(z))) # two-sided test

    p1, z1 = pcs.tests.rayleigh(circ)
    p2, z2 = pcs.tests.rayleigh(linear_circ)

    if (p1 > 0.5) | (p2 > 0.5):
        # Thismeans at least one of our variables may be a uniform distribution
        rho, pval = pcs.descriptive.corrcc_uniform(circ, linear_circ, ci=ci)
    else:
        rho, pval = pcs.descriptive.corrcc(circ, linear_circ, ci=ci)  # circ-circ-corr

    # Assign the correct sign to rho
    if sl < 0:
        rho = -np.abs(rho)
    else:
        rho = np.abs(rho)

    return rho, pval, sl, offs


@numba.jit(nopython=True)
def pcorrelate(t, u, bins):
    """
    From:https://github.com/OpenSMFS/pycorrelate

    Compute correlation of two arrays of discrete events (Point-process).

    The input arrays need to be values of a point process, such as
    photon arrival times or positions. The correlation is efficiently
    computed on an arbitrary array of lag-bins. As an example, bins can be
    uniformly spaced in log-space and span several orders of magnitudes.
    (you can use :func:`make_loglags` to creat log-spaced bins).
    This function implements the algorithm described in
    `(Laurence 2006) <https://doi.org/10.1364/OL.31.000829>`__.

    Arguments:
        t (array): first array of "points" to correlate. The array needs
            to be monothonically increasing.
        u (array): second array of "points" to correlate. The array needs
            to be monothonically increasing.
        bins (array): bin edges for lags where correlation is computed.
        normalize (bool): if True, normalize the correlation function
            as typically done in FCS using :func:`pnormalize`. If False,
            return the unnormalized correlation function.

    Returns:
        Array containing the correlation of `t` and `u`.
        The size is `len(bins) - 1`.
    """
    nbins = len(bins) - 1

    # Array of counts (histogram)
    counts = np.zeros(nbins, dtype=np.int64)

    # For each bins, imin is the index of first `u` >= of each left bin edge
    imin = np.zeros(nbins, dtype=np.int64)
    # For each bins, imax is the index of first `u` >= of each right bin edge
    imax = np.zeros(nbins, dtype=np.int64)

    # For each ti, perform binning of (u - ti) and accumulate counts in Y
    for ti in t:
        for k, (tau_min, tau_max) in enumerate(zip(bins[:-1], bins[1:])):
            if k == 0:
                j = imin[k]
                # We start by finding the index of the first `u` element
                # which is >= of the first bin edge `tau_min`
                while j < len(u):
                    if u[j] - ti >= tau_min:
                        break
                    j += 1

            imin[k] = j
            if imax[k] > j:
                j = imax[k]
            while j < len(u):
                if u[j] - ti >= tau_max:
                    break
                j += 1
            imax[k] = j
            # Now j is the index of the first `u` element >= of
            # the next bin left edge
        counts += imax - imin
    G = counts / np.diff(bins)
    return G

def fast_acf(counts, width, bin_width, cut_peak=True):
    """
    Super fast ACF function relying on numba <3
    :param cut_peak:
    :param counts:
    :param width:
    :param bin_width:
    :return:
    """

    n_b = int(np.ceil(width / bin_width))  # Num. edges per side
    # Define the edges of the bins (including rightmost bin)
    bins = np.linspace(-width, width, 2 * n_b, endpoint=True)
    temp = pcorrelate(counts, counts, np.split(bins, 2)[1])
    acf = np.ones(bins.shape[0] - 1)
    acf[0:temp.shape[0]] = np.flip(temp)
    acf[temp.shape[0]] = temp[0]
    acf[temp.shape[0] + 1:] = temp

    if cut_peak:
        acf[np.nanargmax(acf)] = np.sort(acf)[-2]

    return acf, bins