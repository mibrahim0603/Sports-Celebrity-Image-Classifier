import numpy as np
import pywt
import cv2


def w2d(img, mode='haar', level=1):
    """
    Apply a 2D wavelet transform to an image and reconstruct it using
    only the high-frequency (detail) coefficients, effectively acting
    as a high-pass / edge-detection filter.

    Parameters
    ----------
    img   : np.ndarray
        Input image as a NumPy array with shape (H, W) or (H, W, C).
        Expected dtype is uint8 with pixel values in [0, 255].
    mode  : str, optional
        Wavelet type supported by PyWavelets (default: 'haar').
    level : int, optional
        Decomposition level (default: 1).

    Returns
    -------
    np.ndarray
        Reconstructed grayscale image (uint8) retaining only detail
        (high-frequency) information.

    Raises
    ------
    ValueError
        If `img` is not a valid NumPy array or has an unsupported shape.
    """
    # --- Input validation ---------------------------------------------------
    if not isinstance(img, np.ndarray):
        raise ValueError("Input 'img' must be a NumPy array.")
    if img.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2-D or 3-D array, got shape {img.shape}.")

    # Work on a copy so the caller's array is never mutated
    imArray = img.copy()

    # --- Convert to grayscale -----------------------------------------------
    if imArray.ndim == 3:
        if imArray.shape[2] == 4:          # RGBA → RGB first
            imArray = cv2.cvtColor(imArray, cv2.COLOR_RGBA2GRAY)
        elif imArray.shape[2] == 3:
            imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(
                f"Unsupported channel count: {imArray.shape[2]}. "
                "Expected 3 (RGB) or 4 (RGBA)."
            )
    # If already 2-D (grayscale), use as-is

    # --- Normalise to [0, 1] float32 ----------------------------------------
    imArray = np.float32(imArray) / 255.0

    # --- Wavelet decomposition -----------------------------------------------
    coeffs = pywt.wavedec2(imArray, wavelet=mode, level=level)

    # Zero out the approximation (low-frequency) coefficients so only the
    # high-frequency detail bands remain after reconstruction.
    coeffs_H = list(coeffs)
    coeffs_H[0] = np.zeros_like(coeffs_H[0])   # cleaner than in-place *=0

    # --- Reconstruction ------------------------------------------------------
    imArray_H = pywt.waverec2(coeffs_H, wavelet=mode)

    # Clip to [0, 1] before scaling – reconstruction can produce tiny
    # out-of-range values due to floating-point arithmetic.
    imArray_H = np.clip(imArray_H, 0.0, 1.0)

    # Scale back to [0, 255] and convert to uint8
    imArray_H = np.uint8(imArray_H * 255)

    # Trim any extra row/column that pywt may add during reconstruction
    imArray_H = imArray_H[:imArray.shape[0], :imArray.shape[1]]

    return imArray_H
