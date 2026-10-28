from __future__ import annotations

import numpy as np
import cv2

try:
    import colour
    HAS_COLOUR = True
except ImportError:
    HAS_COLOUR = False

# Fallback arrays (CIE 1931 2-degree CMFs and D65, 380-780 nm, 5 nm steps)
_CMFS_X = [0.001368, 0.002236, 0.004243, 0.00765, 0.01431, 0.02319, 0.04351, 0.07763, 0.13438, 0.21477, 0.2839, 0.3285, 0.34828, 0.34806, 0.3362, 0.3187, 0.2908, 0.2511, 0.19536, 0.1421, 0.09564, 0.05795001, 0.03201, 0.0147, 0.0049, 0.0024, 0.0093, 0.0291, 0.06327, 0.1096, 0.1655, 0.2257499, 0.2904, 0.3597, 0.4334499, 0.5120501, 0.5945, 0.6784, 0.7621, 0.8425, 0.9163, 0.9786, 1.0263, 1.0567, 1.0622, 1.0456, 1.0026, 0.9384, 0.8544499, 0.7514, 0.6424, 0.5419, 0.4479, 0.3608, 0.2835, 0.2187, 0.1649, 0.1212, 0.0874, 0.0636, 0.04677, 0.0329, 0.0227, 0.01584, 0.01135916, 0.008110916, 0.005790346, 0.004109457, 0.002899327, 0.00204919, 0.001439971, 0.0009999493, 0.0006900786, 0.0004760213, 0.0003323011, 0.0002348261, 0.0001661505, 0.000117413, 8.307527e-05, 5.870652e-05, 4.150994e-05]
_CMFS_Y = [3.9e-05, 6.4e-05, 0.00012, 0.000217, 0.000396, 0.00064, 0.00121, 0.00218, 0.004, 0.0073, 0.0116, 0.01684, 0.023, 0.0298, 0.038, 0.048, 0.06, 0.0739, 0.09098, 0.1126, 0.13902, 0.1693, 0.20802, 0.2586, 0.323, 0.4073, 0.503, 0.6082, 0.71, 0.7932, 0.862, 0.9148501, 0.954, 0.9803, 0.9949501, 1.0, 0.995, 0.9786, 0.952, 0.9154, 0.87, 0.8163, 0.757, 0.6949, 0.631, 0.5668, 0.503, 0.4412, 0.381, 0.321, 0.265, 0.217, 0.175, 0.1382, 0.107, 0.0816, 0.061, 0.04458, 0.032, 0.0232, 0.017, 0.01192, 0.00821, 0.005723, 0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.00074, 0.00052, 0.0003611, 0.0002492, 0.0001719, 0.00012, 8.48e-05, 6e-05, 4.24e-05, 3e-05, 2.12e-05, 1.499e-05]
_CMFS_Z = [0.006450001, 0.01055, 0.02005001, 0.03621, 0.06785001, 0.1102, 0.2074, 0.3713, 0.6456, 1.03905, 1.3856, 1.62296, 1.74706, 1.7826, 1.77211, 1.7441, 1.6692, 1.5281, 1.28764, 1.0419, 0.8129501, 0.6162, 0.46518, 0.3533, 0.272, 0.2123, 0.1582, 0.1117, 0.07824999, 0.05725001, 0.04216, 0.02984, 0.0203, 0.0134, 0.008749999, 0.005749999, 0.0039, 0.002749999, 0.0021, 0.0018, 0.001650001, 0.0014, 0.0011, 0.001, 0.0008, 0.0006, 0.00034, 0.00024, 0.00019, 0.0001, 5e-05, 3e-05, 2e-05, 1e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_D65_SD = [49.9755, 52.3118, 54.6482, 68.7015, 82.7549, 87.1204, 91.486, 92.4589, 93.4318, 90.057, 86.6823, 95.7736, 104.865, 110.936, 117.008, 117.41, 117.812, 116.336, 114.861, 115.392, 115.923, 112.367, 108.811, 109.082, 109.354, 108.578, 107.802, 106.296, 104.79, 106.239, 107.689, 106.047, 104.405, 104.225, 104.046, 102.023, 100.0, 98.1671, 96.3342, 96.0611, 95.788, 92.2368, 88.6856, 89.3459, 90.0062, 89.8026, 89.5991, 88.6489, 87.6987, 85.4936, 83.2886, 83.4939, 83.6992, 81.863, 80.0268, 80.1207, 80.2146, 81.2462, 82.2778, 80.281, 78.2842, 74.0027, 69.7213, 70.6652, 71.6091, 72.979, 74.349, 67.9765, 61.604, 65.7448, 69.8856, 72.4863, 75.087, 69.3398, 63.5927, 55.0054, 46.4182, 56.6118, 66.8054, 65.0941, 63.3828]

def get_cmf_d65() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get wavelengths, CMF values (X, Y, Z), and D65 spectral distribution."""
    if HAS_COLOUR:
        try:
            cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(colour.SpectralShape(380, 780, 5))
            d65 = colour.SDS_ILLUMINANTS["D65"].copy().align(colour.SpectralShape(380, 780, 5))
            return cmfs.wavelengths, cmfs.values, d65.values
        except Exception:
            pass

    wavelengths = np.arange(380, 785, 5, dtype=float)
    cmfs_values = np.column_stack((_CMFS_X, _CMFS_Y, _CMFS_Z))
    d65_values = np.array(_D65_SD, dtype=float)
    return wavelengths, cmfs_values, d65_values

def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to linear and then sRGB, with clipping."""
    # Standard 3x3 matrix for XYZ to sRGB
    M = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    linear_srgb = xyz @ M.T
    linear_srgb = np.clip(linear_srgb, 0.0, 1.0)
    
    # sRGB gamma correction
    mask = linear_srgb <= 0.0031308
    srgb = np.empty_like(linear_srgb)
    srgb[mask] = 12.92 * linear_srgb[mask]
    srgb[~mask] = 1.055 * (linear_srgb[~mask] ** (1 / 2.4)) - 0.055
    return srgb

def generate_michel_levy_lut(save_path: str) -> dict:
    """
    Generate Michel-Lévy colour lookup table and save to NPZ.
    
    Parameters
    ----------
    save_path : str
        Path to save the .npz file
        
    Returns
    -------
    dict
        Dictionary containing gamma_nm, rgb, and lab arrays.
    """
    wavelengths, cmfs, d65 = get_cmf_d65()
    
    gammas = np.arange(0, 3002, 2, dtype=float)
    
    # 1. Compute spectrum S(lambda)
    # Transmission spectrum at 45 deg between crossed polars: D65(lambda) * sin^2(pi * Gamma / lambda)
    S = np.zeros((len(gammas), len(wavelengths)))
    for i, w in enumerate(wavelengths):
        S[:, i] = d65[i] * np.sin(np.pi * gammas / w)**2
        
    # 2. Convert to XYZ
    # Sum over wavelengths (trapezoidal integration is common, but basic sum works for 5nm steps if normalized)
    X = (S * cmfs[:, 0]).sum(axis=1)
    Y = (S * cmfs[:, 1]).sum(axis=1)
    Z = (S * cmfs[:, 2]).sum(axis=1)
    
    # Normalize so D65 alone gives Y=1
    Y_d65 = (d65 * cmfs[:, 1]).sum()
    X /= Y_d65
    Y /= Y_d65
    Z /= Y_d65
    xyz = np.column_stack((X, Y, Z))
    
    # 3. Convert XYZ to sRGB
    srgb = xyz_to_srgb(xyz)
    srgb_uint8 = np.round(srgb * 255.0).astype(np.uint8)
    
    # 4. Convert sRGB to CIELAB
    # Using cv2 BGR2Lab on uint8 is requested as an option, but for best precision, we can use float32
    # cv2.cvtColor requires BGR format for cv2.COLOR_BGR2Lab
    # Since we have RGB, we use cv2.COLOR_RGB2Lab
    srgb_float32 = srgb.astype(np.float32).reshape(-1, 1, 3)
    lab_float32 = cv2.cvtColor(srgb_float32, cv2.COLOR_RGB2Lab).reshape(-1, 3)
    
    out_dict = {
        "gamma_nm": gammas,
        "rgb": srgb_uint8,
        "lab": lab_float32
    }
    
    np.savez(save_path, **out_dict)
    return out_dict

def lookup_gamma(xpl_rel_lab: np.ndarray, lut: dict, max_gamma: float = 2000.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the nearest retardation Gamma by minimizing dE76 for given Lab values.
    
    Parameters
    ----------
    xpl_rel_lab : np.ndarray
        Array of CIELAB values (float32), shape [..., 3].
    lut : dict
        The lookup table dictionary generated by generate_michel_levy_lut.
    max_gamma : float
        Maximum gamma value to consider for the lookup (default 2000).
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        gamma_nm: Array of nearest gamma values, matching shape of xpl_rel_lab [..., 0].
        gamma_dE: Array of corresponding dE76 distances.
    """
    gammas = lut["gamma_nm"]
    lab_lut = lut["lab"]
    
    # Filter by max_gamma
    mask = gammas <= max_gamma
    gammas_filtered = gammas[mask]
    lab_filtered = lab_lut[mask]
    
    # Reshape inputs for broadcasting
    orig_shape = xpl_rel_lab.shape
    if orig_shape[-1] != 3:
        raise ValueError("Last dimension of xpl_rel_lab must be 3")
        
    flat_lab = xpl_rel_lab.reshape(-1, 3)
    
    # Handle NaNs: compute only on valid pixels
    valid_mask = ~np.isnan(flat_lab).any(axis=1)
    
    out_gamma = np.full(flat_lab.shape[0], np.nan)
    out_de = np.full(flat_lab.shape[0], np.nan)
    
    if np.any(valid_mask):
        valid_lab = flat_lab[valid_mask]
        
        # Vectorized Euclidean distance (dE76)
        # diffs shape: [N_valid, M_lut, 3]
        diffs = valid_lab[:, np.newaxis, :] - lab_filtered[np.newaxis, :, :]
        de = np.sqrt(np.sum(diffs**2, axis=2))
        
        min_idx = np.argmin(de, axis=1)
        min_de = de[np.arange(len(min_idx)), min_idx]
        
        out_gamma[valid_mask] = gammas_filtered[min_idx]
        out_de[valid_mask] = min_de
        
    return out_gamma.reshape(orig_shape[:-1]), out_de.reshape(orig_shape[:-1])
