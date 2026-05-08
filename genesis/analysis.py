import numpy as np
from scipy.special import jv


def MingXie(
    Kgen,
    ku,
    gamma,
    rel_e_spread,
    I,
    beam_size,
    normemittance,
    wavelength,
    iwityp,
):
    """Return Ming Xie FEL estimates using the notebook's established API."""
    IA = 17.045e3 # Alfven current in A
    fc = (
        jv(0, Kgen**2 / (2 * (1 + Kgen**2))) - jv(1, Kgen**2 / (2 * (1 + Kgen**2)))
        if iwityp == 0
        else 1.0
    )
    rho = (1 / gamma) * (((Kgen * fc / (4 * ku * beam_size)) ** 2) * I / IA) ** (1 / 3)
    Lg1d = 1 / (2 * ku * rho * np.sqrt(3))

    rayleigh_length = 4 * np.pi * beam_size**2 / wavelength
    nd = Lg1d / rayleigh_length

    geoemittance = normemittance / gamma
    beta = beam_size**2 / geoemittance
    ne = (Lg1d / beta) * (4 * np.pi * geoemittance / wavelength)
    ng = 2 * Lg1d * ku * rel_e_spread

    a = [
        0,
        0.45,
        0.57,
        0.55,
        1.6,
        3,
        2,
        0.35,
        2.9,
        2.4,
        51,
        0.95,
        3,
        5.4,
        0.7,
        1.9,
        1140,
        2.2,
        2.9,
        3.2,
    ]
    mx_correction = (
        a[1] * nd**a[2]
        + a[3] * ne**a[4]
        + a[5] * ng**a[6]
        + a[7] * ne**a[8] * ng**a[9]
        + a[10] * nd**a[11] * ng**a[12]
        + a[13] * nd**a[14] * ne**a[15]
        + a[16] * nd**a[17] * ne**a[18] * ng**a[19]
    )
    Lg = Lg1d * (mx_correction + 1)

    # Pbeam[TW] = E[GeV] * I[kA]
    Pbeam = gamma * 0.511e-3 * I * 1e-3 * 1e12
    Pnoise = gamma * 0.511e6 * 3e8 * 1.6e-19 * rho**2 / wavelength
    Psat = 1.6 * rho * (Lg1d / Lg) ** 2 * Pbeam

    return {
        "rho": rho,
        "Lg": Lg,
        "Lg1d": Lg1d,
        "Pbeam": Pbeam,
        "Pnoise": Pnoise,
        "Psat": Psat,
        "fc": fc,
        "RayleighLength": rayleigh_length,
        "beta": beta,
        "geoemittance": geoemittance,
        "eta_d": nd,
        "eta_e": ne,
        "eta_gamma": ng,
        "Lambda": mx_correction,
    }
