# lee_filter.py

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

def lee_filter_xarray(data, kernel_size=3):
    """
    Apply the Lee Speckle Filter directly to a 2D or 3D xarray.DataArray.
    
    Parameters:
    ----------
    data : xarray.DataArray
        Input data (e.g., 2D or 3D array with dimensions like time, y, x)
    kernel_size : int, optional
        Size of the moving window (e.g., 3 for a 3x3 kernel). Default is 3.

    Returns:
    -------
    xarray.DataArray
        Filtered xarray.DataArray with reduced speckle noise.
    """
    def apply_filter(image):
        local_mean = uniform_filter(image, size=kernel_size)
        local_mean_sq = uniform_filter(image**2, size=kernel_size)
        local_variance = local_mean_sq - local_mean**2
        noise_variance = np.var(image) * 0.1  # assume 10% noise
        weights = local_variance / (local_variance + noise_variance)
        return local_mean + weights * (image - local_mean)

    # Apply the Lee filter across xarray structure
    return xr.apply_ufunc(
        apply_filter,
        data,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[data.dtype],
    )

##------------------------------------------------------

# smi_calculation.py

import xarray as xr

def calculate_smi_ratio(S1):
    """
    Calculate the Soil Moisture Index (SMI) using the ratio of VV and VH backscatter coefficients.
    This index highlights differences in soil moisture using SAR polarizations.

    Parameters
    ----------
    S1 : xarray.Dataset
        Input dataset containing filtered VV and VH bands (e.g., 'vv_filtered' and 'vh_filtered').

    Returns
    -------
    xarray.DataArray
        Soil Moisture Index (SMI), normalized between -1 and 1, representing relative soil moisture.
    """
    # Ensure required bands exist
    if "vv_filtered" not in S1 or "vh_filtered" not in S1:
        raise KeyError("Dataset must contain 'vv_filtered' and 'vh_filtered' variables.")

    vv_filtered = S1["vv_filtered"]
    vh_filtered = S1["vh_filtered"]

    # Compute SMI using polarization ratio formula
    smi = (vv_filtered - vh_filtered) / (vv_filtered + vh_filtered)

    # Attach metadata
    smi.name = "SMI"
    smi.attrs["long_name"] = "Soil Moisture Index (VV-VH)/(VV+VH)"
    smi.attrs["description"] = "Normalized polarization ratio sensitive to surface soil moisture"
    smi.attrs["units"] = "unitless"

    return smi

#------------------------------------
# smi_vv_only.py

import xarray as xr

def calculate_smi_vv_only(S1, sigma_min, sigma_max):
    """
    Calculate the Soil Moisture Index (SMI) using only the VV polarization backscatter.
    
    This method normalizes VV values to a range from 0 (wettest) to 1 (driest),
    based on minimum and maximum expected VV values (sigma_min, sigma_max).

    Parameters
    ----------
    S1 : xarray.Dataset
        Input dataset containing the 'vv_filtered' DataArray.
    sigma_min : float
        Minimum expected VV backscatter (represents very wet conditions).
    sigma_max : float
        Maximum expected VV backscatter (represents very dry conditions).

    Returns
    -------
    xarray.DataArray
        Normalized Soil Moisture Index (0–1 scale) derived from VV polarization.
    """
    # Ensure the required variable exists
    if "vv_filtered" not in S1:
        raise KeyError("Dataset must contain a 'vv_filtered' variable.")

    vv_filtered = S1["vv_filtered"]

    # Compute normalized SMI using VV band only
    smi_vv_only = (vv_filtered - sigma_min) / (sigma_max - sigma_min)

    # Clip values to valid range [0, 1]
    smi_vv_only = smi_vv_only.clip(0, 1)

    # Add metadata
    smi_vv_only.name = "SMI_VV"
    smi_vv_only.attrs["long_name"] = "Soil Moisture Index (VV-only)"
    smi_vv_only.attrs["description"] = (
        "Normalized VV backscatter index: 0 = wettest, 1 = driest"
    )
    smi_vv_only.attrs["units"] = "unitless"

    return smi_vv_only
