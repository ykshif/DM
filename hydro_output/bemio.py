import numpy as np
import xarray as xr

def radiationIRF(hydro, tEnd=100, nDt=1001, nDw=1001, wMin=None, wMax=None):
    # If wMin or wMax are not specified, set them to the minimum and maximum frequencies from hydro.omega
    if wMin is None:
        wMin = hydro.omega.min()
    if wMax is None:
        wMax = hydro.omega.max()

    # Generate linearly spaced time and frequency arrays
    t = np.linspace(0, tEnd, nDt)
    w = np.linspace(wMin, wMax, nDw)

    # Initialize the radiation impulse response function (ra_K) array with NaNs
    dof_sum = hydro.radiating_dof.size
    ra_K = np.nan * np.zeros((len(t), dof_sum, dof_sum))

    # Calculate the radiation impulse response function for each degree of freedom (DOF) pair
    for i in range(dof_sum):
        for j in range(dof_sum):
            # Interpolate radiation damping data to the frequency grid
            ra_B = np.interp(w, hydro.omega, hydro.radiation_damping[:, i, j])
            # Integrate over frequency to get the radiation impulse response function (ra_K)
            ra_K[:, i, j] = (2/np.pi) * np.trapz(ra_B * np.cos(w[:, np.newaxis] * t), w, axis=1)

    # Update the hydro data structure with the calculated radiation impulse response function
    hydro['ra_K'] = (('time', 'dof1', 'dof2'), ra_K)

    # Compute the infinite frequency added mass for each DOF pair
    ra_Ainf_temp = np.zeros(len(hydro.omega))
    hydro['Ainf'] = (('dof1', 'dof2'), np.zeros((dof_sum, dof_sum)))
    for i in range(dof_sum):
        for j in range(dof_sum):
            # Interpolate added mass data to the frequency grid
            ra_A = np.interp(w, hydro.omega, hydro.added_mass[:,i, j])
            # Retrieve the previously computed ra_K for the current DOF pair
            ra_K = hydro['ra_K'][:, i, j]
            # Calculate the infinite frequency added mass at each discrete frequency point
            for k in range(len(ra_Ainf_temp)):
                ra_Ainf_temp[k] = ra_A[k] + (1. / w[k]) * np.trapz(ra_K * np.sin(w[k] * t), t)
            # Compute the average to get the final value for the infinite frequency added mass
            hydro.Ainf[i, j] = np.mean(ra_Ainf_temp)

    # Append time and frequency data to the hydro structure
    hydro['ra_t'] = t
    hydro['ra_w'] = w

    return hydro
