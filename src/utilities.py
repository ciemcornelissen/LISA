import os
import subprocess
import sys
from torch.utils.data import Dataset

try:
    import toml
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaleido"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spectral"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "toml"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchmetrics"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "memory_profiler"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torcheval"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn"])

from tqdm import tqdm
import spectral as sp
import numpy as np
import torch
import math
import cv2
import pandas as pd
import h5py
import torch.nn as nn
from scipy.signal import savgol_filter, savgol_coeffs  # Still need scipy to get the coefficients
import pickle
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import gc
from torchmetrics.regression import R2Score
from torcheval.metrics import MeanSquaredError
import torch.multiprocessing as mp
import pytz # For timezone handling
brussels_tz = pytz.timezone('Europe/Brussels')
from torch.autograd import Function
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import weight_norm
from ultralytics import YOLO
from numpy.linalg import svd, inv
import wandb
from datetime import datetime
import torch.optim as optim  # Added for schedulers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import itertools
from memory_profiler import profile
import time
import plotly.graph_objects as go
import io
import random
# import umap
import re
import shutil # For safely removing the old model directory
from sklearn.manifold import TSNE

# try:
#     # Set the start method to 'spawn' BEFORE creating DataLoaders with workers
#     mp.set_start_method('spawn', force=True)
#     print("Multiprocessing start method set to 'spawn'.")
# except RuntimeError as e:
#     # Might raise RuntimeError if already set or in an incompatible context
#     print(f"Note: Could not set multiprocessing start method ('spawn'): {e}")
#     # If it was already set to spawn, this is fine.
#     # If it was set to something else by another library, there might be conflicts.
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

sp.config.SpySettings.envi_support_nonlowercase_params = True


class Augmenter(object):
    def __init__(self, augmentation_config: dict):
        self.augmentation_config = augmentation_config

    def _random_flip(self, _t: torch.Tensor):
        d = np.random.randint(0, 4)

        if d == 0:
            return _t
        if d == 1:
            return _t.flip(1)
        if d == 2:
            return _t.flip(2)
        if d == 3:
            return _t.flip([1, 2])

    def _random_noise(self, _t: torch.Tensor):
        _n = torch.normal(0, 1e-1, _t.shape)
        return _t + _n

    def _random_noise_v2(self, _t: torch.Tensor):
        ch, h, w = _t.shape

        _std = _t.std((1, 2)).expand(h, w, ch).permute(2, 0, 1)
        _mean = _t.mean((1, 2)).expand(h, w, ch).permute(2, 0, 1)

        _n = torch.normal(_mean, _std)
        return _t + (0.8 * _n)

    def _cut_off(self, _t: torch.Tensor):
        _cut_off_size = 16
        _center_cut_off = np.random.randint(0, 64, 2)
        _x_0 = int(max(_center_cut_off[0] - (_cut_off_size / 2), 0))
        _y_0 = int(max(_center_cut_off[1] - (_cut_off_size / 2), 0))
        _x_1 = int(min(_center_cut_off[0] + (_cut_off_size / 2), 63))
        _y_1 = int(min(_center_cut_off[1] + (_cut_off_size / 2), 63))

        _t[:, _x_0:_x_1, _y_0:_y_1] = 0
        return _t

    def _nearest_interp2d(self, input, coords):
        """
        2d nearest neighbor interpolation th.Tensor
        """
        # take clamp of coords so they're in the image bounds
        x = torch.clamp(coords[:, :, 0], 0, input.size(1) - 1).round()
        y = torch.clamp(coords[:, :, 1], 0, input.size(2) - 1).round()

        stride = torch.LongTensor(input.stride())
        x_ix = x.mul(stride[1]).long()
        y_ix = y.mul(stride[2]).long()

        input_flat = input.view(input.size(0), -1)

        mapped_vals = input_flat.gather(1, x_ix.add(y_ix))

        return mapped_vals.view_as(input)

    def _bilinear_interp2d(self, input, coords):
        """
        bilinear interpolation in 2d
        """
        x = torch.clamp(coords[:, :, 0], 0, input.size(1) - 2)
        x0 = x.floor()
        x1 = x0 + 1
        y = torch.clamp(coords[:, :, 1], 0, input.size(2) - 2)
        y0 = y.floor()
        y1 = y0 + 1

        stride = torch.LongTensor(input.stride())
        x0_ix = x0.mul(stride[1]).long()
        x1_ix = x1.mul(stride[1]).long()
        y0_ix = y0.mul(stride[2]).long()
        y1_ix = y1.mul(stride[2]).long()

        input_flat = input.view(input.size(0), -1)

        vals_00 = input_flat.gather(1, x0_ix.add(y0_ix))
        vals_10 = input_flat.gather(1, x1_ix.add(y0_ix))
        vals_01 = input_flat.gather(1, x0_ix.add(y1_ix))
        vals_11 = input_flat.gather(1, x1_ix.add(y1_ix))

        xd = x - x0
        yd = y - y0
        xm = 1 - xd
        ym = 1 - yd

        x_mapped = (vals_00.mul(xm).mul(ym) +
                    vals_10.mul(xd).mul(ym) +
                    vals_01.mul(xm).mul(yd) +
                    vals_11.mul(xd).mul(yd))

        return x_mapped.view_as(input)

    def _iterproduct(self, *args):
        return torch.from_numpy(np.indices(args).reshape((len(args), -1)).T)

    def _affine2d(self, x, matrix, mode='bilinear', center=True):
        """
        2D Affine image transform on torch.Tensor

        Arguments
        ---------
        x : torch.Tensor of size (C, H, W)
            image tensor to be transformed
        matrix : torch.Tensor of size (3, 3) or (2, 3)
            transformation matrix
        mode : string in {'nearest', 'bilinear'}
            interpolation scheme to use
        center : boolean
            whether to alter the bias of the transform
            so the transform is applied about the center
            of the image rather than the origin
        Example
        -------
                # >>> import torch
                # >>> x = torch.zeros(2,1000,1000)
                # >>> x[:,100:1500,100:500] = 10
                # >>> matrix = torch.FloatTensor([[1.,0,-50],
                # ...                             [0,1.,-50]])
                # >>> xn = _affine2d(x, matrix, mode='nearest')
                # >>> xb = _affine2d(x, matrix, mode='bilinear')
        """

        if matrix.dim() == 2:
            matrix = matrix[:2, :]
            matrix = matrix.unsqueeze(0)
        elif matrix.dim() == 3:
            if matrix.size()[1:] == (3, 3):
                matrix = matrix[:, :2, :]

        A_batch = matrix[:, :, :2]
        if A_batch.size(0) != x.size(0):
            A_batch = A_batch.repeat(x.size(0), 1, 1)
        b_batch = matrix[:, :, 2].unsqueeze(1)

        # make a meshgrid of normal coordinates
        _coords = self._iterproduct(x.size(1), x.size(2))
        coords = _coords.unsqueeze(0).repeat(x.size(0), 1, 1).float()  # .to(device)

        if center:
            # shift the coordinates so center is the origin
            coords[:, :, 0] = coords[:, :, 0] - (x.size(1) / 2. - 0.5)
            coords[:, :, 1] = coords[:, :, 1] - (x.size(2) / 2. - 0.5)
        # apply the coordinate transformation
        new_coords = coords.bmm(A_batch.transpose(1, 2)) + b_batch.expand_as(coords)

        if center:
            # shift the coordinates back so origin is origin
            new_coords[:, :, 0] = new_coords[:, :, 0] + (x.size(1) / 2. - 0.5)
            new_coords[:, :, 1] = new_coords[:, :, 1] + (x.size(2) / 2. - 0.5)

        # map new coordinates using bilinear interpolation
        if mode == 'nearest':
            x_transformed = self._nearest_interp2d(x.contiguous(), new_coords)
        elif mode == 'bilinear':
            x_transformed = self._bilinear_interp2d(x.contiguous(), new_coords)

        return x_transformed

    def _random_rotate(self, _i):
        """
        rotates between -45° and 45°

        :param _i:
        :return:
        """

        random_degree = np.random.randint(-90, 90)

        theta = math.pi / 180 * random_degree
        rotation_matrix = torch.tensor(
            [[math.cos(theta), -math.sin(theta), 0],
             [math.sin(theta), math.cos(theta), 0],
             [0, 0, 1]], dtype=torch.float32)  # .to(device)
        input_tf = self._affine2d(_i,
                                  rotation_matrix,
                                  center=True,
                                  mode='nearest')
        return input_tf

    def _random_cut(self, x, ratio=1 / 2):
        h, w = x.shape[1:]

        cut_w = int(w * ratio)
        cut_h = int(h * ratio)

        pos_x = np.random.randint(0 - cut_w, w)
        pos_y = np.random.randint(0 - cut_h, h)

        x = x.clone()
        x[:, max(0, pos_x):min(w, pos_x + cut_w), max(0, pos_y):min(h, pos_y + cut_h)] = 0

        return x

    def __call__(self, batch):
        x = batch

        if self.augmentation_config['random_flip']:
            x = self._random_flip(x)
        if self.augmentation_config['random_rotate']:
            x = self._random_rotate(x)
        if self.augmentation_config['random_noise']:
            x = self._random_noise_v2(x)
        if self.augmentation_config['random_cut']:
            x = self._random_cut(x)

        return x


# --- Activation Map  ---
ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,  # Also known as Swish
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'identity': nn.Identity
}
AUGMENTER = Augmenter(
    {'random_flip': False, 'random_rotate': False, 'random_noise': False, 'random_cut': False})


def get_metrics(predicted_value, target_reg):
    R2_metric = R2Score()
    MSE_metric = MeanSquaredError()
    R2_metric.update(predicted_value, target_reg.squeeze())
    r_two = R2_metric.compute()
    R2_metric.reset()
    MSE_metric.update(predicted_value, target_reg.squeeze())
    mse = MSE_metric.compute()
    MSE_metric.reset()
    return r_two, mse


def crop_to_mask(grape, mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    grape_cropped = np.copy(grape[:, row_min:row_max + 1, col_min:col_max + 1])
    mask_cropped = np.copy(mask[row_min:row_max + 1, col_min:col_max + 1])
    return grape_cropped, mask_cropped, (row_min, col_min)


def get_grape_squares(grape, mask, size):
    grape_cropped, mask_cropped, crop_start = crop_to_mask(grape, mask)
    channels, height, width = grape_cropped.shape
    squares = []
    locations = []

    mask_cropped = mask_cropped > 0

    for i in range(height - size + 1):
        for j in range(width - size + 1):
            square_mask = mask_cropped[i:i + size, j:j + size]
            if np.all(square_mask):
                square = grape_cropped[:, i:i + size, j:j + size]
                squares.append(square)
                original_i = crop_start[0] + i
                original_j = crop_start[1] + j
                locations.append((i, j))

    return squares, locations, grape_cropped


def prepare_msc_matrix(mean_spectrum_ref_np: np.ndarray, device=None):
    """
    Precomputes the matrix M = (XᵀX)⁻¹Xᵀ for MSC calculation.

    Args:
        mean_spectrum_ref_np: The reference mean spectrum (numpy array, shape (C,)).
        device: The torch device ('cpu' or 'cuda') to perform calculations on.

    Returns:
        torch.Tensor: The precomputed matrix M (shape (2, C)).
        torch.Tensor: The reference spectrum on the specified device (shape (C,)).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean_spectrum_ref = torch.tensor(mean_spectrum_ref_np, dtype=torch.float32, device=device)
    c = mean_spectrum_ref.shape[0]  # Number of bands/channels

    # Create the design matrix X (shape C, 2)
    ones_col = torch.ones(c, 1, device=device, dtype=torch.float32)
    x_ref_col = mean_spectrum_ref.unsqueeze(1)  # Make it (C, 1)
    X = torch.cat((ones_col, x_ref_col), dim=1)  # Shape (C, 2)

    # Calculate XᵀX (shape 2, 2)
    XTX = X.T @ X

    # Calculate (XᵀX)⁻¹ (shape 2, 2)
    try:
        # Add small value to diagonal for numerical stability before inversion
        # Adjust epsilon as needed
        epsilon = 1e-7
        XTX_stable = XTX + torch.eye(2, device=device, dtype=torch.float32) * epsilon
        XTX_inv = torch.inverse(XTX_stable)
    except torch.linalg.LinAlgError:
        print("Error: XᵀX matrix is singular or near-singular. Cannot invert.")
        print("This might happen if the reference spectrum is constant or nearly constant.")
        print("XᵀX:", XTX)
        return None, mean_spectrum_ref  # Return None to indicate failure

    # Calculate M = (XᵀX)⁻¹Xᵀ (shape 2, C)
    M = XTX_inv @ X.T

    print(f"MSC Matrix M prepared on device: {device}")
    return M.float(), mean_spectrum_ref.float()  # Ensure float32


def apply_msc_tensorized(spectra_batch: torch.Tensor, msc_matrix_M: torch.Tensor, epsilon=1e-7):
    """
    Applies MSC efficiently to a batch of spectra using the precomputed matrix.

    Args:
        spectra_batch: Tensor of spectra (shape B, C, H, W or N, C).
                       Will be flattened to (N, C) internally.
        msc_matrix_M: The precomputed MSC matrix M = (XᵀX)⁻¹Xᵀ (shape 2, C).
        epsilon: Small value to add to slope denominator for stability.

    Returns:
        torch.Tensor: The MSC-corrected spectra in the original shape.
    """
    if msc_matrix_M is None:
        print("Warning: MSC matrix M is None. Returning original batch.")
        return spectra_batch

    original_shape = spectra_batch.shape
    device = spectra_batch.device
    dtype = spectra_batch.dtype
    original_ndim = spectra_batch.ndim  # Store original ndim

    if spectra_batch.ndim == 3:  # (B, C, H, W)
        c, h, w = original_shape
        spectra_flat = spectra_batch.permute(1, 2, 0).reshape(-1, c)  # (B*H*W, C)
    elif original_ndim == 2:  # Already (N, C)
        spectra_flat = spectra_batch
        n, c = original_shape
    elif original_ndim == 1:  # Single spectrum (C,)
        c = original_shape[0]
        spectra_flat = spectra_batch.unsqueeze(0)  # Add batch dim -> (1, C)
    else:
        raise ValueError("Input spectra_batch must be 1D (C), 2D (N, C), or 4D (B, C, H, W)")

    if msc_matrix_M.shape[1] != c:
        raise ValueError(f"Mismatch between bands in spectra ({c}) and MSC matrix ({msc_matrix_M.shape[1]})")

        # Ensure M is on the same device
    msc_matrix_M = msc_matrix_M.to(device=device, dtype=dtype)

    # Calculate β_batch = M @ Y_batchᵀ
    # Y_batchᵀ has shape (C, N) where N = B*H*W or N or 1
    beta_batch = msc_matrix_M @ spectra_flat.T  # Shape (2, N)

    # Extract intercepts (a) and slopes (b)
    intercepts_a = beta_batch[0, :]  # Shape (N,)
    slopes_b = beta_batch[1, :]  # Shape (N,)

    # --- Robust Divisor Calculation ---
    # Add epsilon to the absolute value to prevent division by ~zero, while keeping the sign
    divisor = slopes_b.sign() * (torch.abs(slopes_b) + epsilon)
    divisor = divisor.unsqueeze(1)  # Reshape to (N, 1) for broadcasting

    # Apply correction: y_corr = (y - a) / b
    # Reshape a to (N, 1) for broadcasting with y (N, C)
    corrected_spectra_flat = (spectra_flat - intercepts_a.unsqueeze(1)) / divisor  # Shape (N, C)

    # Reshape back to original
    if spectra_batch.ndim == 3:
        corrected_spectra = corrected_spectra_flat.reshape(h, w, c).permute(2, 0, 1)  # (B, C, H, W)
    elif original_ndim == 2:  # ndim == 2
        corrected_spectra = corrected_spectra_flat  # Already (N, C)
    elif original_ndim == 1:  # ndim == 1
        corrected_spectra = corrected_spectra_flat.squeeze(0)  # Remove added batch dim -> (C,)

    return corrected_spectra.to(dtype)  # Ensure original dtype


# def apply_msc_tensorized(spectra_batch: torch.Tensor, msc_matrix_M: torch.Tensor, epsilon=1e-7):
#     """
#     Applies MSC efficiently to a batch of spectra using the precomputed matrix.
#
#     Args:
#         spectra_batch: Tensor of spectra (shape B, C, H, W or N, C).
#                        Will be flattened to (N, C) internally.
#         msc_matrix_M: The precomputed MSC matrix M = (XᵀX)⁻¹Xᵀ (shape 2, C).
#         epsilon: Small value to add to slope denominator for stability.
#
#     Returns:
#         torch.Tensor: The MSC-corrected spectra in the original shape.
#     """
#     if msc_matrix_M is None:
#         print("Warning: MSC matrix M is None. Returning original batch.")
#         return spectra_batch
#
#     original_shape = spectra_batch.shape
#     device = spectra_batch.device
#     dtype = spectra_batch.dtype
#
#     if spectra_batch.ndim == 4: # (B, C, H, W)
#         b, c, h, w = original_shape
#         spectra_flat = spectra_batch.permute(0, 2, 3, 1).reshape(-1, c) # (B*H*W, C)
#     elif spectra_batch.ndim == 2: # Already (N, C)
#         spectra_flat = spectra_batch
#         c = original_shape[1]
#     else:
#         raise ValueError("Input spectra_batch must be 2D (N, C) or 4D (B, C, H, W)")
#
#     if msc_matrix_M.shape[1] != c:
#         raise ValueError(f"Mismatch between bands in spectra ({c}) and MSC matrix ({msc_matrix_M.shape[1]})")
#
#     # Ensure M is on the same device
#     msc_matrix_M = msc_matrix_M.to(device=device, dtype=dtype)
#
#     # Calculate β_batch = M @ Y_batchᵀ
#     # Y_batchᵀ has shape (C, N) where N = B*H*W or N
#     beta_batch = msc_matrix_M @ spectra_flat.T # Shape (2, N)
#
#     # Extract intercepts (a) and slopes (b)
#     intercepts_a = beta_batch[0, :] # Shape (N,)
#     slopes_b = beta_batch[1, :]     # Shape (N,)
#
#     # --- Robust Divisor Calculation ---
#     # Add epsilon to the absolute value to prevent division by ~zero, while keeping the sign
#     divisor = slopes_b.sign() * (torch.abs(slopes_b) + epsilon)  # epsilon is a small value like 1e-7 or 1e-8
#     divisor = divisor.unsqueeze(1)  # Reshape to (N, 1) for broadcasting
#
#     # Apply correction: y_corr = (y - a) / b
#     # Add epsilon to denominator to avoid division by zero/instability
#     # Reshape a and b to (N, 1) for broadcasting with y (N, C)
#     # Apply correction using the safe divisor
#     corrected_spectra_flat = (spectra_flat - intercepts_a.unsqueeze(1)) / divisor
#
#     # Reshape back to original
#     if spectra_batch.ndim == 4:
#         corrected_spectra = corrected_spectra_flat.reshape(b, h, w, c).permute(0, 3, 1, 2) # (B, C, H, W)
#     else: # ndim == 2
#         corrected_spectra = corrected_spectra_flat # Already (N, C)
#
#     return corrected_spectra.to(dtype) # Ensure original dtype

class SavGolFilterGPU(nn.Module):
    """
    Applies Savitzky-Golay filtering along the channel (spectral) dimension
    using GPU-accelerated 1D convolution.

    Assumes input tensor shape like (B, C, H, W) or (C, H, W) or (C,)
    and filters along the C dimension.
    """

    def __init__(self, window_length: int, polyorder: int = 2, deriv: int = 0, delta: float = 1.0, device=None):
        super().__init__()

        if window_length % 2 == 0:
            raise ValueError("window_length must be odd.")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length.")

        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        # self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device if device else torch.device("cpu")

        # Get SavGol coefficients using scipy
        coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)

        # Scipy returns filter centered, Conv1d applies left-aligned. Reverse coeffs.
        coeffs = coeffs[::-1].copy()
        coeffs_tensor = torch.tensor(coeffs, dtype=torch.float32)  # .to(self.device) # Move later

        # Reshape coeffs for nn.Conv1d: (out_channels=1, in_channels/groups=1, kernel_size)
        kernel = coeffs_tensor.view(1, 1, window_length)

        # Create the non-trainable Conv1d layer
        # Padding='same' ensures output length matches input length along the filtered dim
        # Padding mode 'replicate' is often reasonable for spectral boundaries
        self.filter = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=window_length,
            padding='same',  # Output size = Input size (along C dim)
            bias=False,
            padding_mode='replicate'  # Or 'reflect', 'circular'
        )

        # Set the fixed weights
        self.filter.weight = nn.Parameter(kernel, requires_grad=False)

        # Move the filter layer to the target device
        self.filter.to(self.device)
        print(f"SavGol GPU filter initialized on device: {self.device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the filter. Handles different input dimensions.
        Input x shape: (B, C, H, W) or (C, H, W) or (C,)
        Output shape: Matches input shape.
        """
        original_shape = x.shape
        original_device = x.device
        original_dtype = x.dtype
        ndim = x.ndim

        # Ensure input is on the same device as the filter
        x = x.to(self.device, dtype=torch.float32)  # Filter uses float32

        if ndim == 1:  # Input shape (C,)
            c = original_shape[0]
            # Reshape to (N=1, Channels=1, SeqLen=C) for Conv1d
            x_reshaped = x.view(1, 1, c)
        elif ndim == 3:  # Input shape (C, H, W)
            c, h, w = original_shape
            # Reshape to (N=H*W, Channels=1, SeqLen=C)
            x_reshaped = x.permute(1, 2, 0).reshape(-1, 1, c)  # -> (H, W, C) -> (H*W, 1, C)
        elif ndim == 4:  # Input shape (B, C, H, W)
            b, c, h, w = original_shape
            # Reshape to (N=B*H*W, Channels=1, SeqLen=C)
            x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, 1, c)  # -> (B, H, W, C) -> (B*H*W, 1, C)
        else:
            raise ValueError(f"Input tensor ndim must be 1, 3, or 4. Got {ndim}")

        # Apply convolution
        filtered_x_reshaped = self.filter(x_reshaped)  # Output shape (N, 1, C)

        # Reshape back to original structure
        if ndim == 1:
            filtered_x = filtered_x_reshaped.view(c)
        elif ndim == 3:
            filtered_x = filtered_x_reshaped.view(h, w, c).permute(2, 0, 1)  # -> (H, W, C) -> (C, H, W)
        elif ndim == 4:
            filtered_x = filtered_x_reshaped.view(b, h, w, c).permute(0, 3, 1, 2)  # -> (B, H, W, C) -> (B, C, H, W)

        # Return tensor on original device and dtype
        return filtered_x.to(device=original_device, dtype=original_dtype)


class grape_dataLoader(Dataset):
    def __init__(self, Augmentations=AUGMENTER, camera='10', front_or_back=False, preprocess_technique=0,
                 indivdualGrapes=False, zoomIn=False, numberAvg=-1, black_ref=True, white_ref=True, normalize=False,
                 Indexs=False, train=False, scalars=False, patches=False, patchSize=16, fullImage=False,
                 save_data=False, just_save_patchStarts=False, labo=True, sweep=False, sg_window=12, pixelMean=False,
                 fullImage_patches=False, background=False, only_background=False, preLoadData=True, longSweep=False,
                 probBackground=0.5, extendedData=False, transfer=True, autoEncoder=False, extendedDataReferences = False, extendDataJustWhiteRef = False, logsep = False):
        self.file_path_brixValues = '/project_ghent/data/grapes2024/FAIR_10_2024/labo_24102024/Laboresultaten_aciditeit_en_brix.xlsx'
        self.file_path_bunchWeights = '/project_ghent/data/grapes2024/FAIR_10_2024/labo_24102024/Gewichten_druiven_PCFruit.xlsx'
        self.indivdualGrapes = indivdualGrapes
        if not indivdualGrapes:
            self.front_or_back = front_or_back
        else:
            self.front_or_back = 'Front'
        self.segLabels_directorys = [f'/project_ghent/data/grapes2024/labels{" "}FX{camera}{" "}front/',
                                     f'/project_ghent/data/grapes2024/labels_scan/',
                                     f'/project_ghent/data/grapes2024/labels_sweep/']
        self.extendedDataReferences = extendedDataReferences
        self.extendedData = extendedData
        self.autoEncoder = autoEncoder
        self.labo = labo
        self.transfer = transfer
        self.probBackground = probBackground
        self.longSweep = longSweep
        self.background = background
        self.only_background = only_background
        self.pixelMean = pixelMean
        self.sg_window = sg_window
        self.extendDataJustWhiteRef = extendDataJustWhiteRef
        self.fullImage_patches = fullImage_patches
        self.sweep = sweep
        self.logsep = logsep
        self.black_ref = black_ref
        self.white_ref = white_ref
        self.numberAvg = numberAvg
        self.camera = camera
        self.preprocess_technique = preprocess_technique
        self.segLocations = [["", "", "", "", "", "", "", ""] for _ in range(60)]  # moveback 76
        self.domain_ids = []
        self.zoomIn = zoomIn
        self.dataLocations = {}
        self._data_finder()
        self._processLabels()

        self.data = []
        self.labels = []
        self.patchLocations = []
        self.backgrounddata = []
        self.patch_sizes = []
        self.reference_ids = []
        self.whiteReferences = []

        self.augmentor = Augmentations
        self.idToKey = {}
        self._idToKey()
        self.patches = patches
        self.patchSize = patchSize
        self.save_data = save_data
        self.just_save_patchStarts = just_save_patchStarts
        self.extendDataCount = 0
        if preLoadData:
            self._preload_data()
            if extendedData:
                if self.extendedDataReferences and not self.extendDataJustWhiteRef:
                    self.white_ref = True
                    # self.black_ref = True
                    self._preload_data()
                    self.white_ref = False
                    # self.black_ref = True
                elif self.extendDataJustWhiteRef:
                    self.extendDataCount += 1
                    self.white_ref = True
                    # self.black_ref = True
                    self._preload_data()
                    self.white_ref = False
                    # self.black_ref = True
                else:
                    self.labo = bool(extendedData[2])
                    self.sweep = bool(extendedData[3])
                    self._preload_data()
                if self.extendedDataReferences:
                    self.white_ref = True
                    # self.black_ref = True
                    self._preload_data()
                    self.white_ref = False


            if not self.save_data:
                self.Indexs = Indexs
                if self.logsep:
                    self._logsep()
                self.normalize = normalize
                self.train = train
                self.scalers = scalars
                self.scaler_labels_brix = None
                self.scaler_labels_acid = None
                self.scaler_labels_ratio = None
                self.scaler_labels_weights = None
                self.scaler_images = None
                self.scaler_pixel = None
                self.fullImage = fullImage
                self.scaler_images = []
                self._normalize()

                # self.device = torch.device('cuda' if torch.cuda.is_available() and patchSize >= 40 else 'cpu')
                self.device = torch.device('cpu')
                if scalars or isinstance(self.scaler_images, tuple):
                    self.msc_matrix_M, _ = prepare_msc_matrix(self.scaler_images[0] if not scalars else self.scalers[3],
                                                              device=self.device)
                # --- Initialize GPU SavGol Filters (if needed by techniques) ---
                self.savgol_gpu_deriv1 = None
                self.savgol_gpu_deriv2 = None
                self.savgol_device = self.device

                # Instantiate filters only if a SavGol technique might be selected
                savgol_techniques = [2, 3, 4, 5]  # Indices where SavGol is applied
                if any(tech == preprocess_technique for tech in savgol_techniques):
                    # Ensure window is odd
                    if sg_window % 2 == 0: sg_window += 1
                    print(f"Initializing SavGol GPU filters (Win={sg_window})...")
                    try:
                        # Filter for 1st derivative
                        if any(tech in [2, 4] for tech in savgol_techniques):
                            self.savgol_gpu_deriv1 = SavGolFilterGPU(
                                window_length=sg_window, deriv=1, device=self.savgol_device
                            )
                        # Filter for 2nd derivative
                        if any(tech in [3, 5] for tech in savgol_techniques):
                            self.savgol_gpu_deriv2 = SavGolFilterGPU(
                                window_length=sg_window, deriv=2, device=self.savgol_device
                            )
                    except ValueError as e:
                        print(f"Error initializing SavGolFilterGPU: {e}. SavGol filtering might be skipped.")
                        self.savgol_gpu_deriv1 = None
                        self.savgol_gpu_deriv2 = None

    def __len__(self):
        if self.indivdualGrapes:
            return 456
        return len(self.dataLocations) if self.front_or_back == False else len(self.dataLocations) // 2

    def _data_finder(self):
        camera = self.camera
        data = {}
        directories = []
        if self.labo:
            directories.append(f'/project_ghent/data/grapes2024/FAIR_10_2024/labo_24102024/GrapeBunchFX{camera}e/rij6')
            directories.append(f'/project_ghent/data/grapes2024/FAIR_10_2024/labo_24102024/GrapeBunchFX{camera}e/rij11')
        elif self.longSweep:
            directories.append(f'/project_ghent/data/grapes2024/veld_23102024/rij11_sweep_lang')
        elif not self.sweep:
            directories.append(f'/project_ghent/data/grapes2024/veld_23102024/rij6_scan')
        else:
            directories.append(f'/project_ghent/data/grapes2024/veld_23102024/rij6_sweep')
        for cube_directory in directories:
            for dir in tqdm(os.listdir(cube_directory), desc="Processing directories"):
                data_point_dict = {}
                if dir.split('_')[0] == 'Chardonnay' or dir.split('_')[0] == '20241022':
                    capture_path = os.path.join(cube_directory, dir, 'capture')
                    for file in os.listdir(capture_path):
                        file_type = file.split('_')[0]
                        if file.endswith('.hdr'):
                            data_point_dict[f'{file_type}_hdr'] = os.path.join(capture_path, file)
                        if file.endswith('.raw'):
                            data_point_dict[f'{file_type}_img'] = os.path.join(capture_path, file)

                    if dir.split('_')[5] == '':
                        dir = dir.split('_')
                        dir.pop(5)
                        dir = '_'.join(dir)
                    if dir.split('_')[3] != 'R11':
                        oost = 'Oost' if not (self.sweep and not self.labo) else 'oost'
                        if self.longSweep: oost = 'oost'
                        if dir.split('_')[
                            3 if not (self.sweep and not self.labo) and not self.longSweep else 5] == oost:
                            dir_keys = 'East' + '_' + dir.split('_')[
                                                          4 if self.labo or self.sweep or self.longSweep else 2][3:]
                            dir_keys += '_' + dir.split('_')[5] if self.labo else ''
                        else:
                            dir_keys = 'West' + '_' + dir.split('_')[
                                                          4 if self.labo or self.sweep or self.longSweep else 2][3:]
                            dir_keys += '_' + dir.split('_')[5] if self.labo else ''
                    else:
                        if dir.split('_')[4] == 'Oost':
                            dir_keys = dir.split('_')[3] + '_' + 'East' + '_' + str(
                                (int(dir.split('_')[5][3:]) - 2) * 4 + int(dir.split('_')[6])) + '_' + dir.split('_')[7]
                        else:
                            dir_keys = dir.split('_')[3] + '_' + dir.split('_')[4] + '_' + str(
                                (int(dir.split('_')[5][3:]) - 2) * 4 + int(dir.split('_')[6])) + '_' + dir.split('_')[7]
                    data[dir_keys] = data_point_dict
        self.dataLocations = dict(sorted(data.items()))

        getMin = []
        segLabel_dir = self.segLabels_directorys[0 if self.labo else 1]
        if self.sweep:
            segLabel_dir = self.segLabels_directorys[2]
        for dir in tqdm(os.listdir(segLabel_dir), desc="Processing directories"):
            if not dir == '.ipynb_checkpoints':
                getMin.append(int(dir.split('-')[1]))
        getMin = min(getMin)
        for dir in tqdm(os.listdir(segLabel_dir), desc="Processing directories"):
            if not dir == '.ipynb_checkpoints':
                number = int(dir.split('-')[1]) - getMin
                label = int(dir.split('-')[-2]) - 1
                if number >= 60:
                    continue
                if dir.split('-')[-1][:1] == "1":
                    if self.segLocations[number][label] == "":
                        self.segLocations[number][label] = [dir]
                    else:
                        self.segLocations[number][label] = [self.segLocations[number][label][:], dir]
                else:
                    self.segLocations[number][label] = dir

    def _processLabels(self):
        df = pd.read_excel(self.file_path_brixValues)
        df_weights = pd.read_excel(self.file_path_bunchWeights)

        df.replace('te weinig staal', 21, inplace=True)
        brixWest, avg_brixWest = df.iloc[1:, 7], []
        acidWest, avg_acidWest = df.iloc[1:, 8], []
        brixEast, avg_brixEast = df.iloc[1:, 12], []
        acidEast, avg_acidEast = df.iloc[1:, 13], []
        r11_brixEast, r11_avg_brixEast = df.iloc[1:, 2], []
        r11_acidEast, r11_avg_acidEast = df.iloc[1:, 3], []

        weightsWest = df_weights.iloc[:30, 2]
        weightsEast = df_weights.iloc[:30, 1]
        r11_weights = df_weights.iloc[:30, 3]

        for i in range(30):
            avg_brixWest.append(brixWest.iloc[i * 6: (i + 1) * 6].mean())
            avg_acidWest.append(acidWest.iloc[i * 6: (i + 1) * 6].mean())
            avg_brixEast.append(brixEast.iloc[i * 6: (i + 1) * 6].mean())
            avg_acidEast.append(acidEast.iloc[i * 6: (i + 1) * 6].mean())
        for i in range(16):
            r11_avg_brixEast.append(r11_brixEast.iloc[i * 6: (i + 1) * 6].mean())
            r11_avg_acidEast.append(r11_acidEast.iloc[i * 6: (i + 1) * 6].mean())

        avg_brixWest = pd.DataFrame(avg_brixWest)
        avg_acidWest = pd.DataFrame(avg_acidWest)
        avg_brixEast = pd.DataFrame(avg_brixEast)
        avg_acidEast = pd.DataFrame(avg_acidEast)
        r11_avg_brixEast = pd.DataFrame(r11_avg_brixEast)
        r11_avg_acidEast = pd.DataFrame(r11_avg_acidEast)

        df_extendedValues = pd.concat([brixEast, brixWest, r11_brixEast], ignore_index=True)
        df_extendedValues = df_extendedValues.to_frame()
        df_extendedValues = df_extendedValues.rename(columns={0: 'brix'})
        df_extendedValues['acid'] = pd.concat([acidEast, acidWest, r11_acidEast], ignore_index=True)
        df_extendedValues['ratio'] = df_extendedValues['brix'] / df_extendedValues['acid']
        df_extendedValues_avg = pd.concat([avg_brixEast, avg_brixWest, r11_avg_brixEast], ignore_index=True)
        df_extendedValues_avg = df_extendedValues_avg.rename(columns={0: 'brix'})
        df_extendedValues_avg['acid'] = pd.concat([avg_acidEast, avg_acidWest, r11_avg_acidEast], ignore_index=True)
        df_extendedValues_avg['ratio'] = df_extendedValues_avg['brix'] / df_extendedValues_avg['acid']
        df_extendedValues = df_extendedValues.dropna()

        df_extendedWeights = pd.concat([weightsEast, weightsWest, r11_weights], ignore_index=True)
        df_extendedWeights = df_extendedWeights.to_frame()
        df_extendedWeights = df_extendedWeights.rename(columns={0: 'weights'})
        df_extendedWeights = df_extendedWeights.dropna()

        self.df_extendedWeights = df_extendedWeights
        self.df_extendedValues = df_extendedValues
        self.df_extendedValues_avg = df_extendedValues_avg

    def _load_data(self, data_point_path, file_type=0):
        files = ['Chardonnay', 'WHITEREF', 'DARKREF'] if self.labo else ['20241022', 'WHITEREF', 'DARKREF']
        hdr = sp.envi.open(data_point_path[f'{files[file_type]}_hdr'], data_point_path[f'{files[file_type]}_img'])

        image = hdr.load()
        return image
    def _logsep(self):
        if isinstance(self.logsep,bool):
            self.logsep = LOGSEP()

            self.data = []
            self.labels = []
            self.patchLocations = []
            self.backgrounddata = []
            self.patch_sizes = []
            self.reference_ids = []
            self.whiteReferences = []
            self.white_ref = True
            self._preload_data()
            if self.extendedData:
                if self.extendedDataReferences and not self.extendDataJustWhiteRef:
                    self.white_ref = True
                    # self.black_ref = True
                    self._preload_data()
                    self.white_ref = False
                    # self.black_ref = True
                elif self.extendDataJustWhiteRef:
                    self.extendDataCount += 1
                    self.white_ref = True
                    # self.black_ref = True
                    self._preload_data()
                    self.white_ref = False
                    # self.black_ref = True
                else:
                    self.labo = bool(self.extendedData[2])
                    self.sweep = bool(self.extendedData[3])
                    self._preload_data()
                if self.extendedDataReferences:
                    self.white_ref = True
                    # self.black_ref = True
                    self._preload_data()
                    self.white_ref = False
            radiance_trainingData = []
            illumination_trainingData = []
            for i in tqdm(self.Indexs):
                whiteRef = i // 6
                slice_data = self.data[i][:, (self.data[i] != 0)[0]]
                # Update sums and count
                radiance_trainingData.append(np.sum(slice_data, axis=1))
                illumination_trainingData.append(self.whiteReferences[whiteRef])
            radiance_trainingData = np.array(radiance_trainingData).transpose((1,0))
            illumination_trainingData = np.array(illumination_trainingData).transpose((1,0))
            self.logsep.train(illumination_trainingData, radiance_trainingData)
            self.data = []
            self.labels = []
            self.patchLocations = []
            self.backgrounddata = []
            self.patch_sizes = []
            self.reference_ids = []
            self.whiteReferences = []
            self.white_ref = False
            self._preload_data()
            if self.extendedData:
                if self.extendedDataReferences and not self.extendDataJustWhiteRef:
                    self.white_ref = True
                    # self.black_ref = True
                    self._preload_data()
                    self.white_ref = False
                    # self.black_ref = True
                elif self.extendDataJustWhiteRef:
                    self.extendDataCount += 1
                    self.white_ref = True
                    # self.black_ref = True
                    self._preload_data()
                    self.white_ref = False
                    # self.black_ref = True
                else:
                    self.labo = bool(self.extendedData[2])
                    self.sweep = bool(self.extendedData[3])
                    self._preload_data()
                if self.extendedDataReferences:
                    self.white_ref = True
                    # self.black_ref = True
                    self._preload_data()
                    self.white_ref = False
        else:
            print("Using given LOGSEP")
        for i in tqdm(self.Indexs):
            image = self.data[i] #shape(224.height width)
            image_shape = image.shape
            new_image = np.zeros_like(image)
            for y in range(image_shape[1]):
                for x in range(image_shape[2]):
                    new_image[:, y, x] = self.logsep.separate(image[:, y, x])[0]  # Use reflectance
            self.data[i] = new_image

    def whiteAndBlackRef(self, ID, returnWhiteRef = False):
        data_point_path = self.dataLocations[self.idToKey[ID]]
        hsi_image = self._load_data(data_point_path, 0)
        if self.sweep: hsi_image = hsi_image.transpose((1, 0, 2))[::-1]
        if self.labo:
            whiteRef = self._load_data(data_point_path, 1)
            if returnWhiteRef:
                return np.mean(whiteRef, axis=(0, 1))
            whiteRef = np.mean(whiteRef, axis=0)
            white_ref_expanded = np.expand_dims(whiteRef, axis=0)  # Shape: (1, 1024, 224)
        elif not self.longSweep:
            segLabel_dir = self.segLabels_directorys[2 if self.sweep else 1]
            if isinstance(self.segLocations[ID][6], str):
                mask = np.load(
                    f'{segLabel_dir}{self.segLocations[ID][6]}')[
                       :: 1 if self.labo or self.sweep else -1, ::-1 if not self.sweep else 1]
            else:
                mask = np.load(
                    f'{segLabel_dir}{self.segLocations[ID][6][0]}')[
                       :: 1 if self.labo or self.sweep else -1, ::-1 if not self.sweep else 1]
                mask2 = np.load(
                    f'{segLabel_dir}{self.segLocations[ID][6][1]}')[
                        :: 1 if self.labo or self.sweep else -1, ::-1 if not self.sweep else 1]
                mask = np.logical_or(mask, mask2)

            y_coords, x_coords = np.where(mask > 0)
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            if ID == 0 and not self.sweep:
                y_min += 15
            whiteRef = hsi_image[y_min:y_max + 1, :, :]
            whiteRef = np.mean(whiteRef, axis=0)  # Average over spatial dimensions
            white_ref_expanded = np.expand_dims(whiteRef, axis=0)  # Shape: (1, 1024, 224)
            # hsi_image = hsi_image[:,x_min:x_max + 1, :]
            if self.pixelMean or self.sweep:
                whiteRef = hsi_image[y_min:y_max + 1, x_min:x_max + 1, :]
                whiteRef = np.mean(whiteRef, axis=(0, 1))  # Average over spatial dimensions
                white_ref_expanded = np.expand_dims(whiteRef, axis=(0, 1))  # Shape: (1, 1024, 224)
                if returnWhiteRef:
                    return white_ref_expanded
        darkRef = self._load_data(data_point_path, 2)
        darkRef = np.mean(darkRef, axis=0)
        black_ref_expanded = np.expand_dims(darkRef, axis=0)  # Shape: (1, 1024, 224)
        if self.sweep: black_ref_expanded = black_ref_expanded.transpose((1, 0, 2))[::-1]
        if not self.black_ref:
            black_ref_expanded = 0  # Shape: (827, 1024, 224)
        hsi_image_corrected = hsi_image - black_ref_expanded
        hsi_image_corrected = np.clip(hsi_image_corrected, 0, None)
        if self.white_ref:
            whiteRef_corrected = white_ref_expanded - black_ref_expanded
        else:
            whiteRef_corrected = 1
        hsi_image_corrected = np.divide(hsi_image_corrected, whiteRef_corrected, dtype=np.float32)
        if hsi_image_corrected.shape[1] == 1024 and self.labo:
            hsi_image_corrected = hsi_image_corrected[:, 100:900]
        elif self.labo:
            hsi_image_corrected = hsi_image_corrected[:, 100:500]
        del hsi_image
        return hsi_image_corrected

    def _idToKey(self):
        for i in tqdm(range(152 if self.labo else (60 if not self.longSweep else 2))):
            name = list(self.dataLocations.keys())[i]
            if name.split('_')[-1] == self.front_or_back or self.front_or_back == False or not self.labo:
                if name.split('_')[0] == 'East':
                    ID = (int(name.split('_')[1]) - 1) * 2
                if name.split('_')[0] == 'West':
                    ID = (int(name.split('_')[1]) - 1) * 2 + 60
                if name.split('_')[0] == 'R11':
                    ID = (int(name.split('_')[2]) - 1) * 2 + 120
                if self.labo:
                    ID += int(name.split('_')[3 if name.split('_')[0] == 'R11' else 2] == 'Back')
                ID = ID // 2 if self.front_or_back or not self.labo else ID
                self.idToKey[ID] = name

    def toRGB(self, ID, longSweep_mask = False):
        if isinstance(ID, int):
            data_point = self.whiteAndBlackRef(ID)
        else:
            data_point = ID
        if isinstance(longSweep_mask,list):
            if len(longSweep_mask)== 3:
                data_point = np.copy(data_point[:, longSweep_mask[1]:longSweep_mask[2]])
            else:
                mask = np.load(
                    f'/project_ghent/data/grapes2024/veld_23102024/Rij11_locationGrapeBunches/task-{425 if longSweep_mask[0] == 1 else 426}-annotation-{395 if longSweep_mask[0] == 1 else 396}-by-1-tag-{longSweep_mask[1]}-0.npy')
                y_coords, x_coords = np.where(mask > 0)
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                data_point = np.copy(data_point[y_min:y_max, x_min:x_max])
        data_point = data_point[:, :, [114, 58, 20]]
        histogram, bin_edges = np.histogram(data_point, bins=10)
        scale = bin_edges[-1]
        data_point = data_point / scale
        data_point = np.clip(data_point, 0, 1)
        image_RGB = np.sqrt(data_point)
        return image_RGB

    def _preload_data(self):
        if self.indivdualGrapes and self.patches and self.background and self.save_data:
            self.data = [None for _ in range(456)]
            self.patchLocations = [None for _ in range(456)]
            for i in tqdm(range(60)):  # moveback 76
                if i != 53 or self.sweep:
                    data_point = self.whiteAndBlackRef(i)[:, :, :]
                    data_point = data_point.transpose((2, 0, 1))
                    # if self.sweep: data_point = data_point.transpose((0,2,1))[:,::-1]
                    for j in range(6):
                        if not self.labo:
                            if isinstance(self.segLocations[i][7], str):
                                mask = np.load(
                                    f'{self.segLabels_directorys[0 if self.labo else (2 if self.sweep else 1)]}/{self.segLocations[i][7]}')[
                                       :: 1 if self.labo or self.sweep else -1, ::-1 if not self.sweep else 1]
                            else:
                                mask = np.load(
                                    f'{self.segLabels_directorys[0 if self.labo else (2 if self.sweep else 1)]}/{self.segLocations[i][7][0]}')[
                                       :: 1 if self.labo or self.sweep else -1, ::-1 if not self.sweep else 1]
                                mask2 = np.load(
                                    f'{self.segLabels_directorys[0 if self.labo else (2 if self.sweep else 1)]}/{self.segLocations[i][7][1]}')[
                                        :: 1 if self.labo or self.sweep else -1, ::-1 if not self.sweep else 1]
                                mask = np.logical_or(mask, mask2)

                            y_coords, x_coords = np.where(mask > 0)
                            x_min, x_max = x_coords.min(), x_coords.max()
                            y_min, y_max = y_coords.min(), y_coords.max()
                        background_patches = []
                        c, h, w = data_point.shape
                        ps = self.patchSize
                        max_y = h - ps
                        max_x = w - ps

                        # --- Optimization: Find valid coordinates efficiently ---
                        # Option 1 (Simpler, potentially slow for large images/many patches): Random Sampling
                        while len(background_patches) < 7:
                            y = np.random.randint(0, max_y + 1)
                            x = np.random.randint(0, max_x + 1)

                            if not self.labo and ((x < x_min - ps) or (x > x_max) or
                                                  (y < y_min - ps) or (y > y_max)):
                                patch_data = np.copy(data_point[:, y:y + ps, x:x + ps])
                                background_patches.append(patch_data)
                            if self.labo and ((x < w // 10) or (x > w - w // 10) or
                                              (y < h // 10) or (y > h - h // 10)):
                                patch_data = np.copy(data_point[:, y:y + ps, x:x + ps])
                                background_patches.append(patch_data)
                        grape_ID = i * 6 + j
                        self.data[grape_ID] = background_patches
                        if not self.labo: del mask
                    del data_point
                    gc.collect()
            with open(
                    f'/project_ghent/data/grapes2024/datasets/background_patches_{"" if self.labo else "_L" + str(False)}_{self.camera}_W{self.white_ref}_B{self.black_ref}{"" if self.labo else "_P" + str(self.pixelMean)}{"" if not self.sweep else "_S" + str(self.sweep)}.pkl',
                    'wb') as hf:
                pickle.dump(self.data, hf)
        elif not self.indivdualGrapes and not self.patches:
            if self.save_data:
                number = 76 if self.front_or_back else 152
                self.data = np.zeros((number if not self.fullImage_patches else 60, 224,
                                      64 if not self.fullImage_patches else self.patchSize,
                                      64 if not self.fullImage_patches else self.patchSize))
                for i in tqdm(range(number if not self.fullImage_patches else 60)):
                    if self.fullImage_patches and i == 53: continue
                    data_point = self.whiteAndBlackRef(i)
                    if self.zoomIn:
                        image = self.toRGB(i)
                        mean_image = np.mean(image[:, :, [1]], axis=2)
                        image = (image - np.min(image)) / (np.max(image) - np.min(image))
                        alpha = 0.55 * np.mean(image) / 0.26
                        image[mean_image < alpha] = 0
                        non_zero_indices = np.argwhere(image > 0)
                        min_y, min_x, _ = non_zero_indices.min(axis=0)
                        max_y, max_x, _ = non_zero_indices.max(axis=0)
                        data_point = data_point[min_y:max_y + 1, min_x:max_x + 1]
                    if self.fullImage_patches:
                        if isinstance(self.segLocations[i][7], str):
                            mask = np.load(
                                f'/project_ghent/data/grapes2024/labels_{"scan" if not self.sweep else "sweep"}/{self.segLocations[i][7]}')[
                                   :: 1 if self.labo or self.sweep else -1,
                                   ::-1 if not self.sweep else 1]
                        else:
                            mask = np.load(
                                f'/project_ghent/data/grapes2024/labels_{"scan" if not self.sweep else "sweep"}/{self.segLocations[i][7][0]}')[
                                   :: 1 if self.labo or self.sweep else -1, ::-1 if not self.sweep else 1]
                            mask2 = np.load(
                                f'/project_ghent/data/grapes2024/labels_{"scan" if not self.sweep else "sweep"}/{self.segLocations[i][7][1]}')[
                                    :: 1 if self.labo or self.sweep else -1, ::-1 if not self.sweep else 1]
                            mask = np.logical_or(mask, mask2)

                        y_coords, x_coords = np.where(mask > 0)
                        x_min, x_max = x_coords.min(), x_coords.max()
                        y_min, y_max = y_coords.min(), y_coords.max()
                        data_point = np.copy(data_point[y_min:y_max + 1, x_min:x_max + 1])
                        del mask
                    data_point = cv2.resize(data_point, dsize=(64 if not self.fullImage_patches else self.patchSize,
                                                               64 if not self.fullImage_patches else self.patchSize),
                                            interpolation=cv2.INTER_CUBIC)
                    final_data_point = np.copy(data_point.transpose((2, 0, 1)))
                    self.data[i] = final_data_point
                    del data_point
                    gc.collect()
                if not self.fullImage_patches:
                    with h5py.File(
                            f'/project_ghent/data/grapes2024/bunchesProcessed_{self.camera}_{self.front_or_back}_{self.zoomIn}_W{self.white_ref}_B{self.black_ref}.h5',
                            'w') as hf:
                        hf.create_dataset('data', data=self.data)
                else:
                    with open(
                            f'/project_ghent/data/grapes2024/datasets/bunchesProcessed_{"" if self.labo else "_L" + str(False)}_C{self.camera}_P{self.patchSize}_W{self.white_ref}_B{self.black_ref}{"" if self.labo else "_Pi" + str(self.pixelMean)}{"" if not self.sweep else "_S" + str(self.sweep)}.pkl',
                            'wb') as hf:
                        pickle.dump(self.data, hf)
            else:
                if not self.fullImage_patches:
                    if not self.white_ref:
                        with open(
                                f'/project_ghent/data/grapes2024/bunchesProcessed_{self.camera}_{self.front_or_back}_{self.zoomIn}_W{self.white_ref}_B{self.black_ref}.h5',
                                'rb') as hf:
                            self.data = [hf['data'][0]]
                            for i in tqdm(range(1, hf['data'].shape[0])):
                                self.data.append(hf['data'][i])
                            self.data = np.array(self.data)
                    else:
                        with h5py.File(
                                f'/project_ghent/data/grapes2024/bunchesProcessed_{self.camera}_{self.front_or_back}_{self.zoomIn}.h5',
                                'r') as hf:
                            self.data = [hf['data'][0]]
                            for i in tqdm(range(1, hf['data'].shape[0])):
                                self.data.append(hf['data'][i])
                            self.data = np.array(self.data)
                else:
                    with open(
                            f'/project_ghent/data/grapes2024/datasets/bunchesProcessed_{"" if self.labo else "_L" + str(False)}_C{self.camera}_P{self.patchSize}_W{self.white_ref}_B{self.black_ref}{"" if self.labo else "_Pi" + str(self.pixelMean)}{"" if not self.sweep else "_S" + str(self.sweep)}.pkl',
                            'rb') as hf:
                        if not self.data:
                            self.data = list(pickle.load(hf))[:60]
                        else:
                            self.data.extend(list(pickle.load(hf))[:60])
        elif self.indivdualGrapes and not self.patches:
            if self.save_data:
                self.data = [None for _ in range(456)]
                for i in tqdm(range(76 if self.labo else 60)):
                    if i != 53:
                        data_point = self.whiteAndBlackRef(i)[:, :, :]
                        data_point = data_point.transpose((2, 0, 1))
                        for j in range(6):
                            if isinstance(self.segLocations[i][j], str):
                                mask = np.load(
                                    f'{self.segLabels_directorys[0 if self.labo else 1]}/{self.segLocations[i][j]}')[
                                       :: 1 if self.labo else -1, ::-1]
                            else:
                                mask = np.load(
                                    f'{self.segLabels_directorys[0 if self.labo else 1]}/{self.segLocations[i][j][0]}')[
                                       :: 1 if self.labo else -1, ::-1]
                                mask2 = np.load(
                                    f'{self.segLabels_directorys[0 if self.labo else 1]}/{self.segLocations[i][j][1]}')[
                                        :: 1 if self.labo else -1, ::-1]
                                mask = np.logical_or(mask, mask2)
                            grape = np.copy(data_point[:, mask > 0].squeeze())
                            grape_ID = i * 6 + j
                            self.data[grape_ID] = grape
                            del grape
                        del data_point
                        gc.collect()
                with open(
                        f'/project_ghent/data/grapes2024/datasets/grapesProcessed{"" if self.labo else "_L" + str(False)}_{self.camera}_individualGrapes_W{self.white_ref}_B{self.black_ref}{"" if self.labo else "_P" + str(self.pixelMean)}.pkl',
                        'wb') as hf:
                    pickle.dump(self.data, hf)
            else:
                with open(
                        f'/project_ghent/data/grapes2024/datasets/grapesProcessed{"" if self.labo else "_L" + str(False)}_{self.camera}_individualGrapes_W{self.white_ref}_B{self.black_ref}{"" if self.labo else "_P" + str(self.pixelMean)}.pkl',
                        'rb') as hf:
                    self.data = pickle.load(hf)
        elif self.indivdualGrapes and self.patches:
            if self.save_data:
                self.data = [None for _ in range(456)]
                self.patchLocations = [None for _ in range(456)]
                for i in tqdm(range(60)):  # moveback 76
                    if i != 53 or self.sweep:
                        data_point = self.whiteAndBlackRef(i)[:, :, :]
                        data_point = data_point.transpose((2, 0, 1))
                        for j in range(6):
                            if isinstance(self.segLocations[i][j], str):
                                mask = np.load(
                                    f'{self.segLabels_directorys[0 if self.labo else (2 if self.sweep else 1)]}/{self.segLocations[i][j]}')[
                                       :: 1 if self.labo or self.sweep else -1, ::-1 if not self.sweep else 1]
                            else:
                                mask = np.load(
                                    f'{self.segLabels_directorys[0 if self.labo else (2 if self.sweep else 1)]}/{self.segLocations[i][j][0]}')[
                                       :: 1 if self.labo or self.sweep else -1, ::-1 if not self.sweep else 1]
                                mask2 = np.load(
                                    f'{self.segLabels_directorys[0 if self.labo else (2 if self.sweep else 1)]}/{self.segLocations[i][j][1]}')[
                                        :: 1 if self.labo or self.sweep else -1, ::-1 if not self.sweep else 1]
                                mask = np.logical_or(mask, mask2)
                            mask = (mask / 255.0).astype(data_point.dtype)
                            y_coords, x_coords = np.where(mask > 0)
                            x_min, x_max = x_coords.min(), x_coords.max()
                            y_min, y_max = y_coords.min(), y_coords.max()
                            grape = data_point * mask[None, :, :]
                            squares, patch_starts, grape_cropped = get_grape_squares(grape, mask, self.patchSize)
                            if not patch_starts:
                                patch_starts = [(0, 0), (1, 0), (0, 1), (1, 1)]
                                grape_cropped = np.copy(data_point[:, y_min: y_min + 10, x_min: x_min + 10])
                            grape_ID = i * 6 + j
                            if not self.just_save_patchStarts:
                                self.data[grape_ID] = grape_cropped
                            self.patchLocations[grape_ID] = patch_starts
                            del mask
                            del grape
                        del data_point
                        gc.collect()
                if not self.just_save_patchStarts:
                    with open(
                            f'/project_ghent/data/grapes2024/datasets/grapesProcessed_patches_{"" if self.labo else "_L" + str(False)}_{self.camera}_individualGrapes_W{self.white_ref}_B{self.black_ref}{"" if self.labo else "_P" + str(self.pixelMean)}{"" if not self.sweep else "_S" + str(self.sweep)}.pkl',
                            'wb') as hf:
                        pickle.dump(self.data, hf)
                with open(
                        f'/project_ghent/data/grapes2024/datasets/patchLocations_{"" if self.labo else "_L" + str(False)}_{self.camera}_{self.patchSize}{"" if not self.sweep else "_S" + str(self.sweep)}.pkl',
                        'wb') as hf:
                    pickle.dump(self.patchLocations, hf)
            else:
                with open(
                        f'/project_ghent/data/grapes2024/datasets/grapesProcessed_patches_{"" if self.labo else "_L" + str(False)}_{self.camera}_individualGrapes_W{self.white_ref}_B{self.black_ref}{"" if self.labo else "_P" + str(self.pixelMean)}{"" if not self.sweep else "_S" + str(self.sweep)}.pkl',
                        'rb') as hf:
                    if not self.data:
                        self.data = pickle.load(hf)[:360]
                    else:
                        self.data.extend(pickle.load(hf)[:360])
                    for i in range(360):
                        if not self.extendDataJustWhiteRef:
                            self.domain_ids.append(0 if self.labo else (1 if self.sweep else 1)) # TODO: made quick fix for traning should be set back to (2 if self.sweep else 1)
                            self.reference_ids.append(0 if not self.white_ref else 1)
                        else:
                            self.domain_ids.append(0 if not self.extendDataCount else 1)  # TODO: made quick fix for traning should be set back to (2 if self.sweep else 1)
                            self.reference_ids.append(0 if not self.white_ref else 1)
                patch_size = 20 if self.camera == '10' else 10
                if self.sweep: patch_size = 8
                with open(
                        f'/project_ghent/data/grapes2024/datasets/patchLocations{"" if self.labo else "__L" + str(False)}_{self.camera}_{patch_size if self.labo or self.sweep else 12}{"" if not self.sweep else "_S" + str(self.sweep)}.pkl',
                        'rb') as hf:
                    if not self.patchLocations:
                        self.patchLocations = pickle.load(hf)[:360]
                    else:
                        self.patchLocations.extend(pickle.load(hf)[:360])
                if self.background:
                    with open(
                            f'/project_ghent/data/grapes2024/datasets/background_patches_{"" if self.labo else "_L" + str(False)}_{self.camera}_W{self.white_ref}_B{self.black_ref}{"" if self.labo else "_P" + str(self.pixelMean)}{"" if not self.sweep else "_S" + str(self.sweep)}.pkl',
                            'rb') as hf:
                        if not self.backgrounddata:
                            self.backgrounddata = pickle.load(hf)[:360]
                        else:
                            self.backgrounddata.extend(pickle.load(hf)[:360])
                if self.patch_sizes:
                    self.patch_sizes.extend(self._getpatchSizes(len(self.patch_sizes), len(self.patch_sizes) + 360))
                else:
                    self.patch_sizes = self._getpatchSizes(0, 360)
                if self.logsep:
                    with open(
                            f'/project_ghent/data/grapes2024/datasets/whiteReferences_{"" if self.labo else "_L" + str(False)}_{self.camera}_individualGrapes_W{False}_B{self.black_ref}{"" if self.labo else "_P" + str(self.pixelMean)}{"" if not self.sweep else "_S" + str(self.sweep)}.pkl',
                            'rb') as hf:
                        if not self.whiteReferences:
                            self.whiteReferences = pickle.load(hf)
                            self.whiteReferences.insert(53,np.zeros((224)))
                        else:
                            additionalRerfs = pickle.load(hf)
                            additionalRerfs.insert(53,np.zeros((224)))
                            self.whiteReferences.extend(additionalRerfs)

    def preprocess(self, image):
        # Input is assumed to be a Tensor (C, H, W) or (C, L) or (C,)
        original_dtype = image.dtype
        original_device = image.device  # Keep track of original device

        technique = self.preprocess_technique
        processed_tensor = image  # Start with original tensor

        # Apply SNV or MSC first if needed (Techniques 1, 4, 5, 6)
        # Ensure SNV/MSC methods also work with tensors and return tensors
        if technique == 1:  # SNV
            processed_tensor = self.snv_tensor(processed_tensor)  # Need tensor version of SNV
        elif technique == 6:  # MSC
            if self.msc_matrix_M is not None:  # Check if MSC matrix exists
                # Ensure tensor is float32 and on the correct device for apply_msc_tensorized
                msc_input_tensor = processed_tensor.to(device=self.msc_matrix_M.device, dtype=torch.float32)
                processed_tensor = apply_msc_tensorized(msc_input_tensor, self.msc_matrix_M)
                processed_tensor = processed_tensor.to(original_device, original_dtype)  # Convert back
            else:
                print("Warning: MSC selected but matrix not available. Skipping MSC.")
        elif technique == 4:  # Apply SNV before SavGol Deriv1
            processed_tensor = self.snv_tensor(processed_tensor)
        elif technique == 5:  # Apply SNV before SavGol Deriv2
            processed_tensor = self.snv_tensor(processed_tensor)
        # Apply SavGol GPU filtering if needed (Techniques 2, 3, 4, 5)
        savgol_filter_layer = None
        if technique in [2, 4]:  # 1st Derivative
            savgol_filter_layer = self.savgol_gpu_deriv1
        elif technique in [3, 5]:  # 2nd Derivative
            savgol_filter_layer = self.savgol_gpu_deriv2

        if savgol_filter_layer is not None:
            try:
                # Filter should handle device placement internally now
                processed_tensor = savgol_filter_layer(processed_tensor)
            except Exception as e:
                print(f"Error during GPU SavGol filtering: {e}. Returning previous tensor.")
        elif technique in [2, 3, 4, 5]:
            print(f"Warning: SavGol technique {technique} selected, but GPU filter not initialized.")

        # Ensure output is on the original device and dtype
        return processed_tensor.to(device=original_device, dtype=original_dtype)

        # if self.preprocess_technique == 1:
        #     return self.snv(image)
        # if self.preprocess_technique == 2:
        #     return savgol_filter(image, self.sg_window, polyorder= 2, deriv=1, axis=0)
        # if self.preprocess_technique == 3:
        #     return savgol_filter(image, self.sg_window, polyorder= 2, deriv=2, axis=0)
        # if self.preprocess_technique == 4:
        #     return savgol_filter(self.snv(image), self.sg_window, polyorder= 2, deriv=1, axis=0)
        # if self.preprocess_technique == 5:
        #     return savgol_filter(self.snv(image), self.sg_window, polyorder= 2, deriv=2, axis=0)
        # if self.preprocess_technique == 6:
        #     return apply_msc_tensorized(image.unsqueeze(0), self.msc_matrix_M)[0]
        # else:
        #     return image

    def _getpatchSizes(self, begin, end):
        patch_sizes = []
        for i in range(begin, 456 if self.labo and not self.transfer else end):
            if not i in [318, 319, 320, 321, 322, 323,  678, 679, 680, 681, 682, 683,  1038, 1039,1040,1041,1042,1043, 1398,1399,1400,1401,1402,1403] or (
                    self.labo and not self.transfer) or self.sweep:
                patch_sizes.append(len(self.patchLocations[i]))
                data_point = self.data[i]
                if self.patchSize >= 50:
                    c, h, w = data_point.shape
                    patch_size = 96 if self.camera == '10' else 48
                    if not self.labo:
                        patch_size = 64
                    pad_h_total = max(0, patch_size - h)
                    pad_w_total = max(0, patch_size - w)

                    pad_top = pad_h_total // 2
                    pad_bottom = pad_h_total - pad_top
                    pad_left = pad_w_total // 2
                    pad_right = pad_w_total - pad_left

                    # Apply padding if needed
                    # F.pad format: (pad_left, pad_right, pad_top, pad_bottom)
                    # Applies to last dims first (W, then H)
                    self.data[i] = np.array(
                        F.pad(torch.tensor(data_point), (pad_left, pad_right, pad_top, pad_bottom), mode='constant',
                              value=0.0))
            else:
                patch_sizes.append(0)
        return patch_sizes

    def snv_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Standard Normal Variate scaling per spectrum using PyTorch."""
        if x.ndim == 1:  # Single spectrum (C,)
            mean = torch.mean(x)
            std = torch.std(x)
            std = torch.clamp(std, min=1e-6)  # Avoid division by zero
            return (x - mean) / std
        elif x.ndim >= 3:  # Image-like (..., C, H, W) or (..., C, L) - Scale along C dim
            # Assuming C is the dimension to calculate mean/std over (dim=1 if B, C, H, W)
            # For (C, H, W), dim=0
            # For (C, L), dim=0
            dim_to_reduce = 0 if x.ndim == 3 else 1  # Adjust based on common shapes like (C,H,W) or (B,C,H,W)
            if x.shape[dim_to_reduce] <= 1:  # Cannot compute std dev over dimension of size 1
                print(f"Warning: Cannot apply SNV along dimension {dim_to_reduce} with size {x.shape[dim_to_reduce]}")
                return x

            mean = torch.mean(x, dim=dim_to_reduce, keepdim=True)
            std = torch.std(x, dim=dim_to_reduce, keepdim=True)
            std = torch.clamp(std, min=1e-6)  # Avoid division by zero
            return (x - mean) / std
        else:  # e.g. (C, L)
            print(f"Warning: SNV tensor processing not fully implemented for ndim={x.ndim}")
            return x  # Return unchanged

    def snv(self, input_data):
        output_data = torch.tensor(np.zeros_like(input_data))
        if not self.indivdualGrapes:
            for i in range(input_data.shape[1]):
                for j in range(input_data.shape[2]):
                    output_data[:, i, j] = (input_data[:, i, j] - torch.mean(input_data[:, i, j])) / torch.std(
                        input_data[:, i, j])
        else:
            output_data[:] = (input_data[:] - torch.mean(input_data[:])) / torch.std(
                input_data[:])
        return output_data

    def _normalize(self):
        labels_brix = []
        labels_acid = []
        labels_ratio = []
        labels_weights = []
        images_fitScaler = []
        labels_brix_fitScaler = []
        labels_acid_fitScaler = []
        labels_ratio_fitScaler = []
        labels_weights_fitScaler = []
        total_sum = 0
        total_sq_sum = 0
        total_count = 0
        if self.indivdualGrapes:
            for i in range(456):
                labels_brix.append(self.df_extendedValues.iloc[i, 0])
                labels_acid.append(self.df_extendedValues.iloc[i, 1])
                labels_ratio.append(self.df_extendedValues.iloc[i, 2])
            for i in self.Indexs if self.train and not self.transfer else range(456):
                labels_brix_fitScaler.append(self.df_extendedValues.iloc[i, 0])
                labels_acid_fitScaler.append(self.df_extendedValues.iloc[i, 1])
                labels_ratio_fitScaler.append(self.df_extendedValues.iloc[i, 2])
        else:
            for i in range(60):
                labels_brix.append(self.df_extendedValues_avg.iloc[i, 0])
                labels_acid.append(self.df_extendedValues_avg.iloc[i, 1])
                labels_ratio.append(self.df_extendedValues_avg.iloc[i, 2])
                labels_weights.append(self.df_extendedWeights.iloc[i, 0])
            for i in self.Indexs if self.train and not self.transfer else range(60):
                labels_brix_fitScaler.append(self.df_extendedValues_avg.iloc[i, 0])
                labels_acid_fitScaler.append(self.df_extendedValues_avg.iloc[i, 1])
                labels_ratio_fitScaler.append(self.df_extendedValues_avg.iloc[i, 2])
                labels_weights_fitScaler.append(self.df_extendedWeights.iloc[i, 0])
        if not self.patches and not self.fullImage_patches:
            for i in self.Indexs if self.train else range(456):
                image = self.data[i]
                images_fitScaler.append(np.mean(image, axis=1))
                slice_count = image.shape[1]
                # Update sums and count
                total_sum += np.sum(image, axis=1)
                total_sq_sum += np.sum(np.square(image), axis=1)
                total_count += slice_count
        elif self.fullImage_patches:
            for i in tqdm(self.Indexs if self.train else range(60)):
                slice_data = self.data[i]
                slice_count = slice_data.shape[1] * slice_data.shape[2]
                # Update sums and count
                total_sum += np.sum(slice_data, axis=(1, 2))
                total_sq_sum += np.sum(np.square(slice_data), axis=(1, 2))
                total_count += slice_count
        else:
            for i in tqdm(self.Indexs if self.train else range(456)):
                slice_data = self.data[i][:, (self.data[i] != 0)[0]]
                slice_count = slice_data.shape[1]
                # Update sums and count
                total_sum += np.sum(slice_data, axis=1)
                total_sq_sum += np.sum(np.square(slice_data), axis=1)
                total_count += slice_count
        mean_images = total_sum / total_count

        # Calculate variance and std dev
        # Variance = E[X^2] - (E[X])^2 = (sum(X^2) / n) - mean^2
        variance = (total_sq_sum / total_count) - np.square(mean_images)

        # Handle potential floating point inaccuracies leading to tiny negative variance
        variance[variance < 0] = 0
        std_images = np.sqrt(variance)
        self.scaler_images = (mean_images, std_images)

        if self.scalers:
            self.scaler_labels_brix = self.scalers[0]
            self.scaler_labels_acid = self.scalers[1]
            self.scaler_images = self.scalers[2]
            self.scaler_pixel = self.scalers[2]
            self.scaler_labels_ratio = self.scalers[4]
            if not self.indivdualGrapes: self.scaler_labels_weights = self.scalers[5]
            self.labels_brix = self.scalers[0].transform(np.array(labels_brix).reshape(-1, 1))
            self.labels_acid = self.scalers[1].transform(np.array(labels_acid).reshape(-1, 1))
            self.labels_ratio = self.scalers[4].transform(np.array(labels_ratio).reshape(-1, 1))
            if not self.indivdualGrapes: self.labels_weights = self.scalers[5].transform(
                np.array(labels_weights).reshape(-1, 1))
        else:
            self.scaler_labels_brix = StandardScaler()
            self.scaler_labels_brix.fit_transform(np.array(labels_brix_fitScaler).reshape(-1, 1))
            self.labels_brix = self.scaler_labels_brix.transform(np.array(np.array(labels_brix).reshape(-1, 1)))
            self.scaler_labels_acid = StandardScaler()
            self.scaler_labels_acid.fit_transform(np.array(labels_acid_fitScaler).reshape(-1, 1))
            self.labels_acid = self.scaler_labels_acid.transform(np.array(labels_acid).reshape(-1, 1))
            self.scaler_labels_ratio = StandardScaler()
            self.scaler_labels_ratio.fit_transform(np.array(labels_ratio_fitScaler).reshape(-1, 1))
            self.labels_ratio = self.scaler_labels_ratio.transform(np.array(labels_ratio).reshape(-1, 1))
            if not self.indivdualGrapes:
                self.scaler_labels_weights = StandardScaler()
                self.scaler_labels_weights.fit_transform(np.array(labels_weights_fitScaler).reshape(-1, 1))
                self.labels_weights = self.scaler_labels_weights.transform(np.array(labels_weights).reshape(-1, 1))
            if not self.patches and not self.fullImage_patches:
                self.scaler_pixel = StandardScaler()
                self.scaler_pixel.fit_transform(np.array(images_fitScaler))

    def getLabels(self):
        labels_brix = []
        labels_acid = []
        labels_ratio = []
        for i in self.Indexs if self.train else range(456):
            labels_brix.append(self.df_extendedValues.iloc[i, 0])
            labels_acid.append(self.df_extendedValues.iloc[i, 1])
            labels_ratio.append(self.df_extendedValues.iloc[i, 2])

        labels_brix = np.array(labels_brix)
        labels_acid = np.array(labels_acid)
        labels_ratio = np.array(labels_ratio)
        if self.normalize:
            labels_brix = self.scaler_labels_brix.transform(labels_brix.reshape(-1, 1)).squeeze()
            labels_acid = self.scaler_labels_acid.transform(labels_acid.reshape(-1, 1)).squeeze()
            labels_ratio = self.scaler_labels_ratio.transform(labels_ratio.reshape(-1, 1)).squeeze()
        return labels_brix[self.Indexs], labels_acid[self.Indexs], labels_ratio[self.Indexs]

    def getData(self):
        images = []
        for i in self.Indexs if self.train else range(456):
            if len(self.data[i].shape) > 1:
                images.append(np.mean(self.data[i], axis=1))
            else:
                images.append(self.data[i])
        images = np.array(images)
        return images
    def getData_Labels(self, label = 'brix'):
        images = []
        labels = []
        for i in self.Indexs if self.train else range(456):
            dataPoint = self[i]
            images.append(np.mean(np.array(dataPoint['image']), axis=(1,2)))
            labels.append(dataPoint[label])

        images = np.array(images)
        return images, labels

    def __getitem__(self, ID):
        reconstruction_target = 0
        backgroundOriginal = 1
        if self.fullImage:
            data_point = self.whiteAndBlackRef(ID)
            data_point = data_point.transpose((2, 0, 1))
        else:
            data_point = self.data[ID]
            if self.background:
                backgroundOriginal = np.random.choice([0, 1], 1, p=[self.probBackground, 1 - self.probBackground])[0]
                if not backgroundOriginal or self.only_background:
                    backgroundID = np.random.randint(0, 6)
                    data_point = self.backgrounddata[ID][backgroundID]

        if (not self.indivdualGrapes and not self.fullImage) or self.fullImage_patches:
            data_point = self.augmentor(torch.tensor(data_point))
        if self.indivdualGrapes and not self.patches:
            shuffled_indices = np.random.permutation(data_point.shape[1])
            data_point = data_point[:, shuffled_indices]
            data_point = data_point[:, :self.numberAvg]
            data_point = np.mean(data_point, axis=1)
            if self.normalize:
                data_point = self.scaler_pixel.transform(data_point.reshape(1, -1)).squeeze()
        if self.patches or self.fullImage_patches:
            if not self.fullImage_patches:
                randomlocation = np.random.randint(0, self.patch_sizes[ID])
                randomlocation = self.patchLocations[ID][randomlocation]
                if (self.patchSize <= 50 or self.sweep) and backgroundOriginal:
                    patchSize = self.patchSize if self.camera == '10' else self.patchSize // 2
                    data_point = data_point[:, randomlocation[0]:randomlocation[0] + patchSize,
                                 randomlocation[1]:randomlocation[1] + patchSize]
                if data_point.shape[1] > 8 and self.transfer:
                    data_point = data_point[:, :8, :8]
            if self.autoEncoder:
                reconstruction_target_base = torch.tensor(np.copy(data_point))
                min_val = torch.min(reconstruction_target_base)
                max_val = torch.max(reconstruction_target_base)
                epsilon = 1e-6
                reconstruction_target = (reconstruction_target_base - min_val) / (max_val - min_val + epsilon)
                reconstruction_target = torch.clamp(reconstruction_target, 0, 1)
            if self.normalize:
                data_point = (data_point - self.scaler_images[0][:, np.newaxis, np.newaxis]) / self.scaler_images[1][:,
                                                                                               np.newaxis, np.newaxis]
        data_point = torch.tensor(data_point) if isinstance(data_point, np.ndarray) else data_point
        if self.fullImage:
            image = data_point
        else:
            image = self.preprocess(data_point)
        image = image.clone()
        if self.domain_ids:
            domain_id = self.domain_ids[ID]
            reference_id = self.reference_ids[ID]
        else:
            domain_id = 0
            reference_id = 0
        if ID >= 360 and not self.fullImage_patches:
            ID = ID%360
        if ID >= 60 and self.fullImage_patches:
            ID = ID % 60
        if not self.front_or_back and not self.indivdualGrapes:
            brix_value = self.df_extendedValues_avg.iloc[ID // 2, 0]
            acid_value = self.df_extendedValues_avg.iloc[ID // 2, 1]
            weight_value = self.df_extendedWeights.iloc[ID // 2, 0]
            ratio_value = self.df_extendedValues_avg.iloc[ID // 2, 2]
        elif not self.indivdualGrapes:
            brix_value = self.df_extendedValues_avg.iloc[ID, 0]
            acid_value = self.df_extendedValues_avg.iloc[ID, 1]
            weight_value = self.df_extendedWeights.iloc[ID, 0]
            ratio_value = self.df_extendedValues_avg.iloc[ID, 2]
        else:
            brix_value = self.df_extendedValues.iloc[ID, 0]
            acid_value = self.df_extendedValues.iloc[ID, 1]
            weight_value = self.df_extendedValues.iloc[ID, 0]
            ratio_value = self.df_extendedValues.iloc[ID, 2]
        if self.normalize:
            brix_value = self.labels_brix[ID]
            acid_value = self.labels_acid[ID]
            if not self.indivdualGrapes:
                weight_value = self.labels_weights[ID]
            else:
                weight_value = self.labels_brix[ID]
            ratio_value = self.labels_ratio[ID]
        if self.background:
            if backgroundOriginal == 0:
                if type(brix_value) == np.ndarray:
                    return {'image': image, 'brix': np.array([0.0]), 'acid': np.array([0.0]), 'ratio': np.array([0.0]),
                            'weight': np.array([0.0]), 'cls': np.array([0]),
                            'reconstruction_target': reconstruction_target,
                            'domain_id': torch.tensor(domain_id, dtype=torch.long), 'reference_id': torch.tensor(reference_id, dtype=torch.long)}
                if type(brix_value) == float:
                    return {'image': image, 'brix': 0.0, 'acid': 0.0, 'ratio': 0.0, 'weight': 0.0, 'cls': np.array([0]),
                            'reconstruction_target': reconstruction_target,
                            'domain_id': torch.tensor(domain_id, dtype=torch.long), 'reference_id': torch.tensor(reference_id, dtype=torch.long)}
        data_point = {'image': image, 'brix': brix_value, 'acid': acid_value, 'ratio': ratio_value,
                      'weight': weight_value, 'cls': np.array([1]), 'reconstruction_target': reconstruction_target,
                      'domain_id': torch.tensor(domain_id, dtype=torch.long), 'reference_id': torch.tensor(reference_id, dtype=torch.long)}
        return data_point






class ClassifierNetwork(nn.Module):
    def __init__(self, bands=224, line=False, line_width=180, kernel_count=3):
        super(ClassifierNetwork, self).__init__()
        self.bands = bands
        self.kernel_count = kernel_count

        self.conv = nn.Sequential(
            nn.Conv2d(self.bands, self.bands * self.kernel_count, kernel_size=3, padding=1, groups=self.bands),
            nn.Conv2d(self.bands * self.kernel_count, 25, kernel_size=1),
            nn.ReLU(True),
            nn.AvgPool2d((4 if line_width > 7 else 1, 1) if line else 2),
            nn.BatchNorm2d(25),
            nn.Conv2d(25, 25 * self.kernel_count, kernel_size=3, padding=1, groups=25),
            nn.Conv2d(25 * self.kernel_count, 30, kernel_size=1),
            nn.ReLU(True),
            nn.AvgPool2d((4 if line_width > 7 else 1, 1) if line else 2),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 30 * self.kernel_count, kernel_size=3, padding=1, groups=30),
            nn.Conv2d(30 * self.kernel_count, 50, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(50),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 1),
        )

        self.init_params()

    def init_params(self):
        '''Init layer parameters.'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, _x):
        out = self.conv(_x)
        out = out.view(_x.shape[0], -1)
        out = self.fc(out)
        return out


class MultiTaskMLP(nn.Module):
    def __init__(self, input_size, shared_hidden_sizes, regression_hidden_sizes, use_cnn=False, model_config=None):
        super(MultiTaskMLP, self).__init__()
        self.model_config = model_config
        # Shared layers
        shared_layers = []
        if use_cnn:
            # For CNN, reshape input to (batch_size, 1, input_size)
            self.use_cnn = True
            in_channels = 224
            for hidden_size in shared_hidden_sizes:
                shared_layers.extend([
                    nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(p=model_config['dropout_shared']),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                ])
                in_channels = hidden_size

            # Calculate the flattened size after convolutions
            self.flatten_size = shared_hidden_sizes[-1] * (input_size // (2 ** len(shared_hidden_sizes)))
        else:
            # Original MLP shared layers
            self.use_cnn = False
            in_features = input_size
            for hidden_size in shared_hidden_sizes:
                shared_layers.extend([
                    nn.Linear(in_features, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(p=model_config['dropout_shared'])
                ])
                in_features = hidden_size
            self.flatten_size = shared_hidden_sizes[-1]

        self.shared_layers = nn.Sequential(*shared_layers)

        # Regression head
        reg_layers1 = []
        in_features = self.flatten_size
        for hidden_size in regression_hidden_sizes:
            reg_layers1.extend([
                nn.Dropout(p=model_config['dropout_regression']),
                nn.Linear(in_features, hidden_size),
                nn.ReLU()
            ])
            in_features = hidden_size
        reg_layers1.append(nn.Linear(in_features, 1))
        self.regression_head1 = nn.Sequential(*reg_layers1)

    def forward(self, x):
        if self.use_cnn:
            # Reshape for CNN (batch_size, channels, sequence_length)
            x = x.view(x.size(0), 1, -1)
            shared_output = self.shared_layers(x)
            # Flatten for the fully connected layers
            shared_output = shared_output.view(shared_output.size(0), -1)
        else:
            # Original MLP forward
            x = x.view(x.size(0), -1)
            shared_output = self.shared_layers(x)

        regression_output1 = self.regression_head1(shared_output)

        return regression_output1


# --- Helper Modules (Positional Encoding, Attention Pooling) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Original PE is (max_len, d_model). For batch_first=True, we need (1, max_len, d_model)
        # Or adapt during forward pass based on input shape. Let's keep original and adapt in forward.
        # pe = pe.unsqueeze(0).transpose(0, 1) # Shape (max_len, 1, d_model) for non-batch_first
        self.register_buffer('pe', pe)  # Shape (max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] (if batch_first=True)
               or shape [seq_len, batch_size, embedding_dim] (if batch_first=False)
        """
        # Assuming x is (batch_size, seq_len, d_model) due to batch_first=True common use
        # Add positional encoding to the input tensor.
        # self.pe is (max_len, d_model). We need slice (seq_len, d_model) and unsqueeze for batch.
        pe_to_add = self.pe[:x.size(1), :].unsqueeze(0)  # Shape: (1, seq_len, d_model)
        x = x + pe_to_add.to(x.device)
        return self.dropout(x)


class AttentionPooling1D(nn.Module):
    def __init__(self, num_channels):
        super(AttentionPooling1D, self).__init__()
        self.attention_scorer = nn.Linear(num_channels, 1)

    def forward(self, x):
        # x shape: (batch_size, num_channels, seq_len)
        x_permuted = x.permute(0, 2, 1)  # (batch_size, seq_len, num_channels)
        attention_scores = self.attention_scorer(x_permuted)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # Softmax over seq_len
        # Weighted sum: (batch_size, seq_len, num_channels) * (batch_size, seq_len, 1) -> sum over seq_len
        weighted_sum = torch.sum(x_permuted * attention_weights, dim=1)  # (batch_size, num_channels)
        return weighted_sum


# --- Model Variants ---

# Model 1: CNN + Attention Pooling
class CNN_AttentionPool(nn.Module):
    def __init__(self, input_size, shared_hidden_sizes, regression_hidden_sizes, model_config):
        super().__init__()
        self.model_config = model_config
        shared_layers = []
        in_channels = 1
        current_size = input_size
        for hidden_size in shared_hidden_sizes:
            shared_layers.extend([
                nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(p=model_config['dropout_shared']),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ])
            in_channels = hidden_size
            current_size = current_size // 2  # Keep track of sequence length reduction

        self.shared_layers = nn.Sequential(*shared_layers)
        final_cnn_channels = shared_hidden_sizes[-1]
        self.attention_pool = AttentionPooling1D(final_cnn_channels)

        # Regression head
        reg_layers = []
        in_features = final_cnn_channels  # Input to regression is output of attention pooling
        for hidden_size in regression_hidden_sizes:
            reg_layers.extend([
                nn.Dropout(p=model_config['dropout_regression']),
                nn.Linear(in_features, hidden_size),
                nn.ReLU()
            ])
            in_features = hidden_size
        reg_layers.append(nn.Linear(in_features, 1))
        self.regression_head = nn.Sequential(*reg_layers)

    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.view(x.size(0), 1, -1)  # (batch_size, 1, input_size)
        shared_output = self.shared_layers(x)  # (batch_size, channels, reduced_seq_len)
        pooled_output = self.attention_pool(shared_output)  # (batch_size, channels)
        regression_output = self.regression_head(pooled_output)
        return regression_output


# Model 2: CNN + Self-Attention Layer
class CNN_SelfAttention(nn.Module):
    def __init__(self, input_size, shared_hidden_sizes, regression_hidden_sizes, model_config,
                 num_attention_heads=4):  # Added attention heads
        super().__init__()
        self.model_config = model_config
        shared_layers = []
        in_channels = 1
        current_size = input_size
        for hidden_size in shared_hidden_sizes:
            shared_layers.extend([
                nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(p=model_config['dropout_shared']),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ])
            in_channels = hidden_size
            current_size = current_size // 2

        self.shared_layers = nn.Sequential(*shared_layers)
        final_cnn_channels = shared_hidden_sizes[-1]
        self.final_cnn_channels = final_cnn_channels

        # Positional Encoding for the sequence length after CNN
        self.pos_encoder = PositionalEncoding(final_cnn_channels, dropout=model_config['dropout_shared'],
                                              max_len=current_size + 1)

        # Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=final_cnn_channels,
            num_heads=num_attention_heads,
            dropout=model_config.get('dropout_attention', 0.1),  # Add dropout_attention to config
            batch_first=True  # Expect input as (batch, seq_len, features)
        )
        self.norm1 = nn.LayerNorm(final_cnn_channels)

        # Regression head - input is still final_cnn_channels after aggregation
        reg_layers = []
        in_features = final_cnn_channels
        for hidden_size in regression_hidden_sizes:
            reg_layers.extend([
                nn.Dropout(p=model_config['dropout_regression']),
                nn.Linear(in_features, hidden_size),
                nn.ReLU()
            ])
            in_features = hidden_size
        reg_layers.append(nn.Linear(in_features, 1))
        self.regression_head = nn.Sequential(*reg_layers)

    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.view(x.size(0), 1, -1)  # (batch_size, 1, input_size)
        cnn_output = self.shared_layers(x)  # (batch_size, channels, reduced_seq_len)

        # Prepare for MHA: (batch_size, reduced_seq_len, channels)
        attn_input = cnn_output.permute(0, 2, 1)

        # Add positional encoding
        attn_input = self.pos_encoder(attn_input)

        # Apply Self-Attention + Residual + Norm
        attn_output, _ = self.self_attention(attn_input, attn_input, attn_input)
        attn_output = self.norm1(attn_input + attn_output)  # Add & Norm

        # Aggregate - Example: Mean Pooling over sequence length
        aggregated_output = attn_output.mean(dim=1)  # (batch_size, channels)

        regression_output = self.regression_head(aggregated_output)
        return regression_output


# Model 3: Transformer Encoder with Patch Embedding
class TransformerRegressor(nn.Module):
    def __init__(self, input_size, regression_hidden_sizes, model_config,
                 d_model=16, nhead=8, num_encoder_layers=2, dim_feedforward=256,
                 patch_size=8):  # Added Transformer HParams
        super().__init__()
        self.model_config = model_config
        d_model = model_config['transformer.d_model']
        nhead = model_config['transformer.n_heads']
        num_encoder_layers = model_config['transformer.n_layers']
        dim_feedforward = model_config['transformer.d_ff']
        patch_size = model_config['transformer.patch_size']

        # 1. Patch Embedding
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        # Calculate sequence length after patching
        self.seq_len = (input_size // patch_size)
        if input_size % patch_size != 0:
            print(
                f"Warning: input_size {input_size} not perfectly divisible by patch_size {patch_size}. Adjust logic if needed.")
            # You might want to add padding or adjust patch/stride

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=model_config.get('dropout_posEncoding', 0),
                                              max_len=self.seq_len + 1)  # +1 for potential CLS token if added later

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=model_config.get('dropout_attention', 0),
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. Aggregation (Using mean pooling here)
        reg_in_features = d_model

        # 5. Regression Head
        reg_layers = []
        in_features = reg_in_features
        for hidden_size in regression_hidden_sizes:
            reg_layers.extend([
                nn.Dropout(p=model_config.get('dropout_regression', 0)),
                nn.Linear(in_features, hidden_size),
                nn.ReLU()
            ])
            in_features = hidden_size
        reg_layers.append(nn.Linear(in_features, 1))
        self.regression_head = nn.Sequential(*reg_layers)

        self.d_model = d_model

    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        # print(x, flush=True)
        # 1. Patch Embedding
        x = self.patch_embed(x)  # (batch_size, d_model, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, d_model)
        # print(x, flush=True)
        # 2. Add Positional Encoding
        x = self.pos_encoder(x)
        # print(x, flush=True)
        # 3. Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        # 4. Aggregate (Mean pooling over sequence length)
        aggregated_output = transformer_output.mean(dim=1)  # (batch_size, d_model)

        # 5. Regression Head
        regression_output = self.regression_head(aggregated_output)
        return regression_output


class SpectralPredictor(nn.Module):
    def __init__(self, model_config):
        """
        Initializes the Spectral Predictor model.

        Args:
            model_config (dict): A dictionary containing model hyperparameters.
                Expected keys:
                - model_type (str): 'mlp', 'cnn1d', or 'cnn2d'.
                - num_bands (int): Number of spectral bands in the input data.
                - patch_height (int): Height of the input patch.
                - patch_width (int): Width of the input patch.
                - shared_cnn_channels (list): List of output channels for each shared CNN layer.
                                              (Used for 'cnn1d' and 'cnn2d').
                - shared_cnn_kernels (list): List of kernel sizes for each shared CNN layer.
                                             (Use int for cnn1d, tuple (kh, kw) for cnn2d).
                - shared_cnn_pools (list): List of pooling kernel sizes after each CNN layer.
                                           (Use int for cnn1d, tuple (ph, pw) for cnn2d).
                                           Use 1 or (1,1) for no pooling.
                - shared_mlp_sizes (list): List of hidden layer sizes for shared MLP layers.
                                           (Used for 'mlp' type, and potentially after CNNs
                                            if specified, though usually just flattening).
                - regression_hidden_sizes (list): List of hidden layer sizes for the regression head.
                - dropout_shared (float): Dropout rate for shared layers.
                - dropout_regression (float): Dropout rate for regression head layers.
                - use_batchnorm (bool): Whether to use BatchNorm layers in CNNs.
        """
        super(SpectralPredictor, self).__init__()
        self.model_config = model_config
        self.model_type = model_config['model_type']
        self.num_bands = model_config['num_bands']
        self.patch_height = model_config['patch_height']
        self.patch_width = model_config['patch_width']

        shared_layers_list = []

        # --- Build Shared Layers based on model_type ---
        if self.model_type == 'mlp':
            self.input_flattened_size = self.num_bands * self.patch_height * self.patch_width
            in_features = self.input_flattened_size
            for hidden_size in model_config.get('shared_mlp_sizes', []):  # Allow empty list for direct regression
                shared_layers_list.append(nn.Linear(in_features, hidden_size))
                shared_layers_list.append(nn.ReLU())
                shared_layers_list.append(nn.Dropout(p=model_config['dropout_shared']))
                in_features = hidden_size
            self.flatten_size = in_features  # Size feeding into regression head

        elif self.model_type == 'cnn1d':
            # Treat spectral dimension as sequence length, spatial as channels (flattened)
            # Input expected: (B, H, W, C) -> permute to (B, H*W, C) -> Conv1d needs (B, Channels, Seq_Len)
            # So, input becomes (B, H*W, num_bands) - treating H*W as channels. Less common.
            # OR treat spectral as channels, spatial flattened as sequence. More common.
            # Input expected: (B, H, W, C) -> reshape to (B, C, H*W) for Conv1d
            in_channels = self.num_bands
            current_seq_len = self.patch_height * self.patch_width

            cnn_channels = model_config['shared_cnn_channels']
            cnn_kernels = model_config['shared_cnn_kernels']
            cnn_pools = model_config['shared_cnn_pools']

            if not (len(cnn_channels) == len(cnn_kernels) == len(cnn_pools)):
                raise ValueError("CNN channels, kernels, and pools lists must have the same length")

            for i, (out_channels, kernel_size, pool_size) in enumerate(zip(cnn_channels, cnn_kernels, cnn_pools)):
                # Ensure padding maintains size or adjust as needed. Assume padding = kernel // 2 for simplicity here.
                # A more robust approach might calculate padding based on desired output size.
                padding = kernel_size // 2
                shared_layers_list.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
                if model_config.get('use_batchnorm', True):
                    shared_layers_list.append(nn.BatchNorm1d(out_channels))
                shared_layers_list.append(nn.ReLU())
                if model_config['dropout_shared'] > 0:
                    shared_layers_list.append(nn.Dropout(p=model_config['dropout_shared']))
                if pool_size > 1:
                    shared_layers_list.append(nn.MaxPool1d(kernel_size=pool_size))
                    # Update sequence length after pooling
                    current_seq_len = current_seq_len // pool_size
                in_channels = out_channels

            # Calculate flatten size after last CNN layer
            self.flatten_size = in_channels * current_seq_len

        elif self.model_type == 'cnn2d':
            # Input expected: (B, H, W, C) -> permute to (B, C, H, W) for Conv2d
            in_channels = self.num_bands
            current_h, current_w = self.patch_height, self.patch_width

            cnn_channels = model_config['shared_cnn_channels']
            cnn_kernels = model_config['shared_cnn_kernels']  # Expect tuples (kh, kw)
            cnn_pools = model_config['shared_cnn_pools']  # Expect tuples (ph, pw)

            if not (len(cnn_channels) == len(cnn_kernels) == len(cnn_pools)):
                raise ValueError("CNN channels, kernels, and pools lists must have the same length")

            for i, (out_channels, kernel_size, pool_size) in enumerate(zip(cnn_channels, cnn_kernels, cnn_pools)):
                # Ensure kernel/pool are tuples
                if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
                if isinstance(pool_size, int): pool_size = (pool_size, pool_size)

                # Simple padding calculation 'same'. For stride > 1 or dilation, this needs adjustment.
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)
                shared_layers_list.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
                if model_config.get('use_batchnorm', True):
                    shared_layers_list.append(nn.BatchNorm2d(out_channels))
                shared_layers_list.append(nn.ReLU())
                if model_config['dropout_shared'] > 0:
                    # Dropout usually applied after activation, sometimes before pooling
                    # Or using nn.Dropout2d if spatial dropout is desired
                    shared_layers_list.append(nn.Dropout(p=model_config['dropout_shared']))  # Or nn.Dropout2d

                if pool_size[0] > 1 or pool_size[1] > 1:
                    shared_layers_list.append(nn.MaxPool2d(kernel_size=pool_size))
                    # Update feature map size after pooling
                    # Note: This integer division assumes stride=pool_size and no fractional sizes
                    current_h = current_h // pool_size[0]
                    current_w = current_w // pool_size[1]
                in_channels = out_channels

            # Calculate flatten size after last CNN layer
            self.flatten_size = in_channels * current_h * current_w

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.shared_layers = nn.Sequential(*shared_layers_list)

        # --- Build Regression Head ---
        reg_layers = []
        in_features = self.flatten_size  # Output size from shared layers
        for hidden_size in model_config['regression_hidden_sizes']:
            reg_layers.append(nn.Dropout(p=model_config['dropout_regression']))  # Dropout often before Linear
            reg_layers.append(nn.Linear(in_features, hidden_size))
            reg_layers.append(nn.ReLU())
            in_features = hidden_size
        # Final output layer
        reg_layers.append(nn.Dropout(p=model_config['dropout_regression']))  # Optional dropout before final layer
        reg_layers.append(nn.Linear(in_features, 1))  # Predict single value (sugar content)
        self.regression_head = nn.Sequential(*reg_layers)

    def _calculate_flatten_size(self):
        """
        Calculates the output size of the shared layers dynamically by passing
        a dummy input tensor through them. More robust than manual calculation.
        """
        # Create a dummy input matching the expected batch input shape
        # Assumes input data is (B, H, W, C)
        dummy_input = torch.zeros(1, self.patch_height, self.patch_width, self.num_bands)

        # Apply necessary preprocessing based on model type
        if self.model_type == 'mlp':
            dummy_input = dummy_input.view(1, -1)  # Flatten
        elif self.model_type == 'cnn1d':
            # Reshape to (B, C, H*W)
            dummy_input = dummy_input.permute(0, 3, 1, 2)  # (B, C, H, W)
            dummy_input = dummy_input.reshape(dummy_input.size(0), self.num_bands, -1)  # (B, C, H*W)
        elif self.model_type == 'cnn2d':
            # Permute to (B, C, H, W)
            dummy_input = dummy_input.permute(0, 3, 1, 2)

        # Pass through shared layers
        self.shared_layers.eval()  # Set to eval mode for deterministic output (esp. dropout/batchnorm)
        with torch.no_grad():
            shared_output = self.shared_layers(dummy_input)
        self.shared_layers.train()  # Set back to train mode

        # Return the flattened size
        return shared_output.view(shared_output.size(0), -1).shape[1]

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, patch_height, patch_width, num_bands).

        Returns:
            torch.Tensor: Regression output tensor with shape (batch_size, 1).
        """
        # --- Input Preprocessing based on Model Type ---
        if self.model_type == 'mlp':
            # Flatten the input: (B, H, W, C) -> (B, H*W*C)
            x = x.view(x.size(0), -1)
            if x.shape[1] != self.input_flattened_size:
                raise ValueError(
                    f"Incorrect input flattened size. Expected {self.input_flattened_size}, got {x.shape[1]}")

        elif self.model_type == 'cnn1d':
            # Reshape for Conv1d: (B, H, W, C) -> (B, C, H*W)
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x = x.reshape(x.size(0), self.num_bands, -1)  # (B, C, H*W)

        elif self.model_type == 'cnn2d':
            # Permute for Conv2d: (B, H, W, C) -> (B, C, H, W)
            x = x.permute(0, 3, 1, 2)

        # --- Shared Layers ---
        shared_output = self.shared_layers(x)

        # --- Flatten ---
        # Flatten the output for the regression head
        shared_output_flat = shared_output.view(shared_output.size(0), -1)

        # --- Regression Head ---
        regression_output = self.regression_head(shared_output_flat)

        return regression_output


# Helper function for learnable positional embeddings
# (Can also use fixed sinusoidal embeddings)
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb



class SpectralPredictorWithTransformer(nn.Module):
    def __init__(self, model_config):
        """
        Initializes the Spectral Predictor model, now including Transformer options.

        Args:
            model_config (dict): A dictionary containing model hyperparameters.
                Includes previous keys plus new ones for transformer:
                - model_type (str): 'mlp', 'cnn1d', 'cnn2d', 'transformer', 'cnn_transformer'.
                - num_bands (int): Number of spectral bands.
                - patch_height (int): Height of the input patch.
                - patch_width (int): Width of the input patch.
                # --- CNN specific (used for cnn1d, cnn2d, cnn_transformer) ---
                - shared_cnn_channels (list): Output channels for CNN layers.
                - shared_cnn_kernels (list): Kernel sizes for CNN layers.
                - shared_cnn_pools (list): Pooling sizes for CNN layers.
                - use_batchnorm (bool): Use BatchNorm in CNNs.
                # --- MLP specific (used for mlp) ---
                - shared_mlp_sizes (list): Hidden sizes for shared MLP layers.
                # --- Transformer specific (used for transformer, cnn_transformer) ---
                - transformer_patch_size (int): Size of smaller patches for tokenization (e.g., 4 means 4x4).
                                                Input H/W (or CNN output H/W) must be divisible by this.
                - transformer_embed_dim (int): Embedding dimension for transformer tokens.
                - transformer_depth (int): Number of transformer encoder layers.
                - transformer_heads (int): Number of attention heads.
                - transformer_mlp_dim (int): Dimension of the MLP head within transformer layers.
                - transformer_dropout (float): Dropout rate within transformer layers.
                - use_cls_token (bool): Whether to use a [CLS] token for aggregation. If False, uses mean pooling.
                # --- Common ---
                - regression_hidden_sizes (list): Hidden sizes for the final regression head.
                - dropout_shared (float): Dropout for shared MLP/CNN layers (not Transformer).
                - dropout_regression (float): Dropout for the final regression head.
        """
        super().__init__()
        self.model_config = model_config
        self.model_type = model_config['model_type']
        self.num_bands = model_config['num_bands']
        self.patch_height = model_config['patch_height']
        self.patch_width = model_config['patch_width']
        self.use_pos_embed = model_config.get('use_pos_embed', True)  # Default to True
        self.use_cls_token = model_config.get('use_cls_token', True)  # Default to True
        self.use_multi_head = model_config.get('use_multi_head', False)  # Default to False
        self.instanceNorm = model_config.get('instanceNorm', False)
        self.AE = model_config.get('AE', False)  # Default to False
        self.conv_dim = model_config.get('conv_dim', 2)
        self.discriminator = model_config.get('discriminator', 0)
        encoder_output_shape = model_config.get('encoder_output_shape', [])
        if encoder_output_shape:
            self.encoder_C_deep = encoder_output_shape[0 if self.conv_dim == 2 else 1]  # Default to 0
            self.encoder_H = encoder_output_shape[1 if self.conv_dim == 2 else 2]  # Default to 0
            self.encoder_W = encoder_output_shape[2 if self.conv_dim == 2 else 3]  # Default to 0

        self.cls = model_config.get('cls', False)  # Default to False
        # --- Build Model Components ---
        self.t_embed_dim = None
        self.t_nheads = None
        self.cnn_backbone = None
        self.patch_embed = None
        self.transformer_encoder = None
        self.pos_embed = None
        self.cls_token = None
        self.shared_layers = None  # For non-transformer models
        self.regression_head = None
        self.t_patch_size = None
        current_h, current_w, current_c = self.patch_height, self.patch_width, self.num_bands
        in_features_regressor = 0
        self.activation_type = model_config.get('activation_type', 'relu').lower()  # Default to relu
        if self.activation_type not in ACTIVATION_MAP:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")
        self.activation_layer = ACTIVATION_MAP[self.activation_type]

        # 1. --- Optional CNN Backbone (for cnn_transformer) ---
        if self.model_type == 'cnn_transformer':
            if not all(k in model_config for k in ['shared_cnn_channels', 'shared_cnn_kernels', 'shared_cnn_pools']):
                raise ValueError("CNN parameters required for cnn_transformer model type.")
            self.cnn_backbone = self._build_cnn_backbone('cnn2d')  # Use 2D CNN for backbone
            # Calculate the output dimensions of the CNN backbone
            current_c, current_h, current_w = self._calculate_cnn_output_dim(self.cnn_backbone)
            print(f"CNN Backbone Output Dim: C={current_c}, H={current_h}, W={current_w}")
 
        # 2. --- Patch Embedding (for transformer types) ---
        if self.model_type in ['transformer', 'cnn_transformer']:
            self.t_patch_size = model_config['transformer_patch_size']
            if current_h % self.t_patch_size != 0 or current_w % self.t_patch_size != 0:
                for new_patch_size in range(self.t_patch_size, 0, -1):
                    if current_h % new_patch_size == 0 and current_w % new_patch_size == 0:
                        self.t_patch_size = new_patch_size
                        print(
                            f"Adjusted patch size to {self.t_patch_size} for input dimensions ({current_h}, {current_w})")
                        break

            # self.num_patches = (current_h // self.t_patch_size) * (current_w // self.t_patch_size)
            num_patches_h = math.floor((current_h - self.t_patch_size) / model_config['transformer_stride']) + 1
            num_patches_w = math.floor((current_w - self.t_patch_size) / model_config['transformer_stride']) + 1
            self.num_patches = num_patches_h * num_patches_w
            patch_dim = current_c * self.t_patch_size * self.t_patch_size  # Flattened size of one sub-patch

            self.t_embed_dim = model_config['transformer_embed_dim']
            self.t_nheads = model_config['transformer_heads']
            if self.t_embed_dim % self.t_nheads != 0:
                print(f"Embedding dimension {self.t_embed_dim} must be divisible by number of heads {self.t_nheads}.")
                self.t_nheads = self.t_embed_dim // 2  # Adjust to be divisible by 2
            # Efficient patching and projection using a Conv2d
            self.patch_embed = nn.Conv2d(current_c, self.t_embed_dim,
                                         kernel_size=self.t_patch_size, stride=model_config['transformer_stride'])

            # --- Positional Embedding & CLS Token ---
            if self.use_cls_token and self.use_pos_embed:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.t_embed_dim))
                # Add 1 to sequence length for CLS token
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.t_embed_dim))
                nn.init.trunc_normal_(self.pos_embed, std=.02)  # Initialize positional embedding
                nn.init.trunc_normal_(self.cls_token, std=.02)
            elif self.use_pos_embed:
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.t_embed_dim))
                nn.init.trunc_normal_(self.pos_embed, std=.02)

            # --- Transformer Encoder ---
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.t_embed_dim,
                nhead=self.t_nheads,
                dim_feedforward=model_config['transformer_mlp_dim'],
                dropout=model_config['transformer_dropout'],
                activation=self.activation_type,  # or 'gelu'
                batch_first=True  # Important: Input shape (B, Seq, Emb)
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=model_config['transformer_depth']
            )
            in_features_regressor = self.t_embed_dim  # Input to regressor is the transformer output dim

        # 3. --- Standard MLP/CNN Shared Layers (for non-transformer types) ---
        elif self.model_type == 'mlp':
            self.input_flattened_size = self.num_bands * self.patch_height * self.patch_width
            in_features = self.input_flattened_size
            shared_layers_list = []
            activation_layer_mlp = self.activation_layer()

            for hidden_size in model_config.get('shared_mlp_sizes', []):
                shared_layers_list.append(nn.Linear(in_features, hidden_size))
                shared_layers_list.append(activation_layer_mlp)
                shared_layers_list.append(nn.Dropout(p=model_config['dropout_shared']))
                in_features = hidden_size
            self.shared_layers = nn.Sequential(*shared_layers_list)
            in_features_regressor = in_features  # Input to regressor is output of last shared layer

        elif self.model_type in ['cnn1d', 'cnn2d']:
            self.shared_layers = self._build_cnn_backbone(self.model_type)
            # Calculate flatten size after CNN layers
            final_channels, final_h, final_w = self._calculate_cnn_output_dim(self.shared_layers)
            if self.model_type == 'cnn1d':
                in_features_regressor = final_channels * final_w  # For 1D CNN output (C, SeqLen)
            else:  # cnn2d
                in_features_regressor = final_channels * final_h * final_w

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # 4. --- Regression Head 1 ---
        reg_layers1 = []
        in_features1 = in_features_regressor
        activation_layer_reg1 = self.activation_layer()

        for hidden_size in model_config['regression_hidden_sizes']:
            reg_layers1.append(nn.Dropout(p=model_config['dropout_regression']))
            reg_layers1.append(nn.Linear(in_features1, hidden_size))
            reg_layers1.append(activation_layer_reg1)
            in_features1 = hidden_size
        reg_layers1.append(nn.Dropout(p=model_config['dropout_regression']))
        reg_layers1.append(nn.Linear(in_features1, 1 if not self.discriminator else 2))
        self.regression_head1 = nn.Sequential(*reg_layers1)

        # 5. --- Regression Head 2 ---
        reg_layers2 = []
        in_features2 = in_features_regressor
        activation_layer_reg2 = self.activation_layer()

        for hidden_size in model_config['regression_hidden_sizes']:
            reg_layers2.append(nn.Dropout(p=model_config['dropout_regression']))
            reg_layers2.append(nn.Linear(in_features2, hidden_size))
            reg_layers2.append(activation_layer_reg2)
            in_features2 = hidden_size
        reg_layers2.append(nn.Dropout(p=model_config['dropout_regression']))
        reg_layers2.append(nn.Linear(in_features2, 1))
        self.regression_head2 = nn.Sequential(*reg_layers2)
        if self.cls:
            # 6. --- Classification Head 1 ---
            cls_layers = []
            in_features1 = in_features_regressor
            activation_layer_cls = self.activation_layer()

            for hidden_size in model_config['cls_hidden_sizes']:
                cls_layers.append(nn.Dropout(p=model_config['dropout_regression']))
                cls_layers.append(nn.Linear(in_features1, hidden_size))
                cls_layers.append(activation_layer_cls)
                in_features1 = hidden_size
            cls_layers.append(nn.Dropout(p=model_config['dropout_regression']))
            cls_layers.append(nn.Linear(in_features1, 2))
            self.cls_head = nn.Sequential(*cls_layers)

    def _build_cnn_backbone(self, cnn_type):
        """Helper to build CNN layers (1D or 2D)."""
        layers = []
        activation_layer = self.activation_layer()  # Instantiate the layer
        if cnn_type == 'cnn1d':
            in_channels = self.num_bands
            current_seq_len = self.patch_height * self.patch_width
            cnn_channels = self.model_config['shared_cnn_channels']
            cnn_kernels = self.model_config['shared_cnn_kernels']
            cnn_pools = self.model_config['shared_cnn_pools']

            for out_channels, kernel_size, pool_size in zip(cnn_channels, cnn_kernels, cnn_pools):
                padding = kernel_size // 2
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
                if self.model_config.get('use_batchnorm', True):
                    batchNorm = nn.BatchNorm1d(out_channels) if not self.instanceNorm else nn.InstanceNorm1d(out_channels)
                    layers.append(batchNorm)
                layers.append(activation_layer)
                if self.model_config['dropout_shared'] > 0:
                    layers.append(nn.Dropout(p=self.model_config['dropout_shared']))
                if pool_size > 1:
                    layers.append(nn.MaxPool1d(kernel_size=pool_size))
                in_channels = out_channels
        elif cnn_type == 'cnn2d':
            in_channels = self.num_bands
            cnn_channels = self.model_config['shared_cnn_channels']
            cnn_kernels = self.model_config['shared_cnn_kernels']  # Expect tuples (kh, kw) or int
            cnn_pools = self.model_config['shared_cnn_pools']  # Expect tuples (ph, pw) or int

            for out_channels, kernel_size, pool_size in zip(cnn_channels, cnn_kernels, cnn_pools):
                if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
                if isinstance(pool_size, int): pool_size = (pool_size, pool_size)
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)

                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
                if self.model_config.get('use_batchnorm', True):
                    batchNorm = nn.BatchNorm2d(out_channels) if not self.instanceNorm else nn.InstanceNorm2d(out_channels)
                    layers.append(batchNorm)
                layers.append(activation_layer)
                if self.model_config['dropout_shared'] > 0:
                    layers.append(nn.Dropout(p=self.model_config['dropout_shared']))  # or Dropout2d
                if pool_size[0] > 1 or pool_size[1] > 1:
                    layers.append(nn.MaxPool2d(kernel_size=pool_size))
                in_channels = out_channels
        else:
            raise ValueError("cnn_type must be 'cnn1d' or 'cnn2d'")

        return nn.Sequential(*layers)

    def _calculate_cnn_output_dim(self, cnn_module):
        """Calculates CNN output C, H, W using a dummy forward pass."""
        cnn_module.eval()  # Important for deterministic output size
        with torch.no_grad():
            # Create dummy input (B=1, C, H, W) or (B=1, C, L)
            if isinstance(cnn_module[0], nn.Conv2d):  # First layer tells us type
                dummy_input = torch.zeros(1, self.num_bands, self.patch_height, self.patch_width)
                output = cnn_module(dummy_input)
                cnn_module.train()
                return output.shape[1], output.shape[2], output.shape[3]  # C, H, W
            elif isinstance(cnn_module[0], nn.Conv1d):
                dummy_input = torch.zeros(1, self.num_bands, self.patch_height * self.patch_width)
                output = cnn_module(dummy_input)
                cnn_module.train()
                return output.shape[1], 1, output.shape[2]  # C, H=1 (dummy), W=SeqLen
            else:
                cnn_module.train()
                raise TypeError("First layer in cnn_module is not Conv1d or Conv2d")

    def forward(self, x):
        """
        Forward pass.
        Input x: (batch_size, patch_height, patch_width, num_bands)
        """
        # --- Input Permutation/Reshaping ---
        # PyTorch CNNs/Transformers usually expect Channels first
        # Target shape: (B, C, H, W) for Conv2D/Transformer patching
        # Target shape: (B, C, L) for Conv1D where L=H*W
        # # Target shape: (B, Features) for MLP
        # if self.model_type in ['cnn2d', 'transformer', 'cnn_transformer']:
        #      # (B, H, W, C) -> (B, C, H, W)
        #      x = x.permute(0, 3, 1, 2)
        if self.AE: x = x.view(x.size(0), self.encoder_C_deep, self.encoder_H, self.encoder_W)

        if self.model_type == 'cnn1d':
            # (B, H, W, C) -> (B, C, H*W)
            # x = x.permute(0, 3, 1, 2)
            x = x.reshape(x.size(0), self.num_bands, -1)
        elif self.model_type == 'mlp':
            # (B, H, W, C) -> (B, H*W*C)
            x = x.view(x.size(0), -1)

        # --- Model Body ---
        shared_output = None  # This will be the input to the regression head

        if self.model_type == 'cnn_transformer':
            # 1. CNN Backbone
            x = self.cnn_backbone(x)  # Output (B, C', H', W')
            # 2. Patch Embedding
            x = self.patch_embed(x)  # Output: (B, embed_dim, num_patches_h, num_patches_w)
            x = x.flatten(2)  # Output: (B, embed_dim, num_patches)
            x = x.transpose(1, 2)  # Output: (B, num_patches, embed_dim) - Transformer expects this
            # 3. Add CLS token and Positional Embedding
            if self.use_cls_token and self.use_pos_embed:
                batch_size = x.shape[0]
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
                x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)
                x = x + self.pos_embed  # Add pos embedding (broadcasting across batch)
            elif self.use_pos_embed:
                x = x + self.pos_embed[:, :self.num_patches, :]  # Add pos embedding
            # 4. Transformer Encoder
            x = self.transformer_encoder(x)  # Output: (B, num_patches+1 or num_patches, embed_dim)
            # 5. Aggregate output
            if self.use_cls_token:
                shared_output = x[:, 0]  # Select the CLS token's output
            else:
                shared_output = x.mean(dim=1)  # Mean pool across sequence dimension

        elif self.model_type == 'transformer':
            # 1. Patch Embedding (applied directly to input)
            x = self.patch_embed(x)  # Output: (B, embed_dim, num_patches_h, num_patches_w)
            x = x.flatten(2)  # Output: (B, embed_dim, num_patches)
            x = x.transpose(1, 2)  # Output: (B, num_patches, embed_dim)
            # 2. Add CLS token and Positional Embedding
            if self.use_cls_token and self.use_pos_embed:
                batch_size = x.shape[0]
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.pos_embed
            elif self.use_pos_embed:
                x = x + self.pos_embed[:, :self.num_patches, :]
            # 3. Transformer Encoder
            x = self.transformer_encoder(x)
            # 4. Aggregate output
            if self.use_cls_token:
                shared_output = x[:, 0]
            else:
                shared_output = x.mean(dim=1)

        elif self.model_type in ['mlp', 'cnn1d', 'cnn2d']:
            # Pass through shared MLP/CNN layers
            x = self.shared_layers(x)
            # Flatten output for regression head
            shared_output = x.view(x.size(0), -1)

        # --- Final Regression Heads ---
        regression_output1 = self.regression_head1(shared_output)
        if self.discriminator: return regression_output1
        if self.cls:
            cls_output = self.cls_head(shared_output)
            regression_output2 = self.regression_head2(shared_output)
            return cls_output, regression_output1, regression_output2
        if self.use_multi_head:
            # Pass through second regression head
            regression_output2 = self.regression_head2(shared_output)
            return regression_output1, regression_output2

        return regression_output1


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Activation Map ---
# Define activation functions (add more if needed)


# --- Helper Modules ---

class PositionalEncoding(nn.Module):
    """ Standard sinusoidal positional encoding """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Keep pe as (max_len, d_model) and adapt in forward
        self.register_buffer('pe', pe)  # Shape (max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] (batch_first=True assumed)
        """
        # self.pe is (max_len, d_model). Slice and unsqueeze for batch compatibility.
        pe_to_add = self.pe[:x.size(1), :].unsqueeze(0)  # Shape: (1, seq_len, d_model)
        x = x + pe_to_add.to(x.device)
        return self.dropout(x)


class AttentionPooling1D(nn.Module):
    """ Attention-based pooling for 1D sequences """

    def __init__(self, num_channels):
        super(AttentionPooling1D, self).__init__()
        # Simple linear layer to compute attention scores for each position
        self.attention_scorer = nn.Linear(num_channels, 1)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, num_channels, seq_len]
        """
        # Permute to (batch_size, seq_len, num_channels) for linear layer
        x_permuted = x.permute(0, 2, 1)
        # Compute scores: (batch_size, seq_len, num_channels) -> (batch_size, seq_len, 1)
        attention_scores = self.attention_scorer(x_permuted)
        # Compute weights: Softmax over the sequence length dimension (dim=1)
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape: (batch_size, seq_len, 1)
        # Apply weights: Element-wise multiply and sum over seq_len
        # (batch_size, seq_len, num_channels) * (batch_size, seq_len, 1) -> sum(dim=1) -> (batch_size, num_channels)
        weighted_sum = torch.sum(x_permuted * attention_weights, dim=1)
        return weighted_sum


# --- Main Unified Model ---

class SpectralPixelPredictor(nn.Module):
    def __init__(self, model_config):
        """
        Initializes a flexible spectral predictor for pixel-level data.

        Args:
            model_config (dict): Dictionary containing model hyperparameters.
                Required keys:
                - input_size (int): Number of spectral bands (features per pixel).
                - model_type (str): Architecture type. Options:
                    'mlp', 'cnn1d', 'cnn_attnpool', 'cnn_selfattn', 'transformer'.
                - regression_hidden_sizes (list): Hidden sizes for the final regression head(s).

                Optional/Conditional keys:
                # --- General ---
                - activation_type (str): Type of activation ('relu', 'gelu', etc.). Default: 'relu'.
                - dropout_regression (float): Dropout rate for regression head(s). Default: 0.0.
                - use_multi_head (bool): If True, creates two identical regression heads. Default: False.

                # --- MLP specific ('mlp') ---
                - shared_mlp_sizes (list): Hidden sizes for MLP layers. Default: [].
                - dropout_shared (float): Dropout rate for shared MLP layers. Default: 0.0.

                # --- CNN specific ('cnn1d', 'cnn_attnpool', 'cnn_selfattn') ---
                - cnn_channels (list): Output channels for CNN layers.
                - cnn_kernels (list): Kernel sizes for CNN layers.
                - cnn_pools (list): Pooling sizes (stride) for MaxPool1d after each CNN layer. Use 1 for no pooling.
                - use_batchnorm (bool): Use BatchNorm1d in CNN layers. Default: True.
                - dropout_shared (float): Dropout rate after activation in CNN layers. Default: 0.0.
                - cnn_global_pooling (str): Pooling after last CNN layer for 'cnn1d'. 'max', 'avg'. Default: 'avg'.

                # --- CNN+SelfAttention specific ('cnn_selfattn') ---
                - num_attention_heads (int): Number of heads for MultiheadAttention. Default: 4.
                - dropout_attention (float): Dropout rate within the self-attention mechanism. Default: 0.1.
                - dropout_pos_encoding (float): Dropout rate for positional encoding. Default: 0.1.

                # --- Transformer specific ('transformer') ---
                - transformer_patch_size (int): Size of patches for tokenization via Conv1d.
                - transformer_d_model (int): Embedding dimension (output channels of patch Conv1d).
                - transformer_nhead (int): Number of attention heads in Transformer Encoder.
                - transformer_nlayers (int): Number of layers in Transformer Encoder.
                - transformer_dim_ff (int): Dimension of the feedforward network in Transformer Encoder layers.
                - transformer_dropout_attn (float): Dropout rate in Transformer Encoder attention/FFN. Default: 0.1.
                - transformer_dropout_pos_encoding (float): Dropout rate for positional encoding. Default: 0.1.
                - use_cls_token (bool): Prepend a learnable CLS token for aggregation. Default: False.
        """
        super().__init__()
        self.model_config = model_config
        self.input_size = model_config['input_size']
        self.model_type = model_config['model_type']
        self.use_multi_head = model_config.get('use_multi_head', False)

        # --- Activation ---
        self.activation_type = model_config.get('activation_type', 'relu').lower()
        if self.activation_type not in ACTIVATION_MAP:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")
        self.activation_layer = ACTIVATION_MAP[self.activation_type]

        # --- Initialize Components ---
        self.shared_backbone = None  # MLP layers or CNN layers
        self.pooling_aggregator = None  # AttentionPooling1D or Global Pooling
        self.pos_encoder = None  # Positional Encoding
        self.transformer_components = nn.ModuleDict()  # For patch_embed, encoder, norm, cls_token etc.
        self.self_attention_components = nn.ModuleDict()  # For MHA, norm etc.
        self.regression_head1 = None
        self.regression_head2 = None

        in_features_regressor = 0  # This will be calculated based on the model type

        # --- Build Backbone based on Model Type ---

        if self.model_type == 'mlp':
            mlp_sizes = model_config.get('shared_mlp_sizes', [])
            dropout_shared = model_config.get('dropout_shared', 0.0)
            self.shared_backbone, in_features_regressor = self._build_mlp_backbone(
                self.input_size, mlp_sizes, dropout_shared
            )

        elif self.model_type in ['cnn1d', 'cnn_attnpool', 'cnn_selfattn']:
            # --- Build CNN Backbone (Common for these types) ---
            if not all(k in model_config for k in ['cnn_channels', 'cnn_kernels', 'cnn_pools']):
                raise ValueError("CNN parameters (cnn_channels, cnn_kernels, cnn_pools) required for CNN-based models.")

            self.shared_backbone, current_channels, current_seq_len = self._build_cnn1d_backbone()
            print(f"CNN Backbone Output Dim: C={current_channels}, L={current_seq_len}")

            # --- Build Specific Components for each CNN variant ---
            if self.model_type == 'cnn1d':
                pooling_type = model_config.get('cnn_global_pooling', 'avg')
                if pooling_type == 'avg':
                    # Adaptive pooling takes care of variable sequence lengths
                    self.pooling_aggregator = nn.AdaptiveAvgPool1d(1)
                elif pooling_type == 'max':
                    self.pooling_aggregator = nn.AdaptiveMaxPool1d(1)
                else:
                    raise ValueError(f"Unsupported cnn_global_pooling type: {pooling_type}")
                in_features_regressor = current_channels  # After pooling, only channels remain

            elif self.model_type == 'cnn_attnpool':
                self.pooling_aggregator = AttentionPooling1D(current_channels)
                in_features_regressor = current_channels  # Output of attention pooling

            elif self.model_type == 'cnn_selfattn':
                d_model = current_channels  # Use CNN output channels as attention dimension
                nhead = model_config.get('num_attention_heads', 4)
                dropout_attn = model_config.get('dropout_attention', 0.1)
                dropout_pos = model_config.get('dropout_pos_encoding', 0.1)

                if d_model % nhead != 0:
                    # Find the largest divisor of d_model <= nhead or a sensible default like 1 or 2
                    valid_heads = [h for h in range(nhead, 0, -1) if d_model % h == 0]
                    if not valid_heads:
                        print(
                            f"Warning: CNN output channels ({d_model}) not divisible by requested heads ({nhead}). Falling back to 1 head.")
                        nhead = 1
                    else:
                        nhead = valid_heads[0]
                        print(
                            f"Warning: CNN output channels ({d_model}) not divisible by requested heads. Adjusted heads to {nhead}.")

                self.pos_encoder = PositionalEncoding(d_model, dropout=dropout_pos, max_len=current_seq_len + 1)
                self.self_attention_components['mha'] = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout_attn,
                    batch_first=True
                )
                self.self_attention_components['norm'] = nn.LayerNorm(d_model)
                # Aggregation will happen in forward (e.g., mean pooling)
                in_features_regressor = d_model

        elif self.model_type == 'transformer':
            # --- Build Transformer Components ---
            if not all(k in model_config for k in
                       ['transformer_patch_size', 'transformer_d_model', 'transformer_nhead', 'transformer_nlayers',
                        'transformer_dim_ff']):
                raise ValueError("Transformer parameters required for transformer model type.")

            p_size = model_config['transformer_patch_size']
            d_model = model_config['transformer_d_model']
            nhead = model_config['transformer_nhead']
            nlayers = model_config['transformer_nlayers']
            dim_ff = model_config['transformer_dim_ff']
            dropout_attn = model_config.get('transformer_dropout_attn', 0.1)
            dropout_pos = model_config.get('transformer_dropout_pos_encoding', 0.1)
            self.use_cls_token = model_config.get('use_cls_token', False)

            if self.input_size % p_size != 0:
                print(
                    f"Warning: input_size ({self.input_size}) not perfectly divisible by transformer_patch_size ({p_size}). Effective sequence length might differ.")
            num_patches = math.ceil(self.input_size / p_size)  # Use ceil to handle non-divisible case

            if d_model % nhead != 0:
                # Find the largest divisor of d_model <= nhead or a sensible default like 1 or 2
                valid_heads = [h for h in range(nhead, 0, -1) if d_model % h == 0]
                if not valid_heads:
                    print(
                        f"Warning: transformer_d_model ({d_model}) not divisible by requested heads ({nhead}). Falling back to 1 head.")
                    nhead = 1
                else:
                    nhead = valid_heads[0]
                    print(
                        f"Warning: transformer_d_model ({d_model}) not divisible by requested heads. Adjusted heads to {nhead}.")

            # 1. Patch Embedding (using Conv1d)
            self.transformer_components['patch_embed'] = nn.Conv1d(1, d_model, kernel_size=p_size, stride=p_size)
            self.num_patches = num_patches

            # 2. CLS Token (Optional)
            if self.use_cls_token:
                # self.transformer_components['cls_token'] = nn.Parameter(torch.zeros(1, 1, d_model)) # <-- REMOVE THIS LINE
                self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # <-- DEFINE HERE
                nn.init.trunc_normal_(self.cls_token, std=.02)
                effective_num_patches = self.num_patches + 1  # Sequence length for pos encoding includes CLS
            else:
                effective_num_patches = self.num_patches

            # 3. Positional Encoding
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout_pos,
                                                  max_len=effective_num_patches + 1)  # Max len accommodates CLS

            # 4. Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout_attn,
                activation='gelu' if self.activation_type == 'gelu' else 'relu',
                batch_first=True
            )
            self.transformer_components['encoder'] = nn.TransformerEncoder(
                encoder_layer,
                num_layers=nlayers
            )
            # Optional LayerNorm after encoder
            self.transformer_components['norm'] = nn.LayerNorm(d_model)

            in_features_regressor = d_model  # Output dimension is d_model

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # --- Build Regression Head(s) ---
        if in_features_regressor <= 0:
            raise ValueError(
                f"Could not determine input features for regression head for model type {self.model_type}.")

        dropout_reg = model_config.get('dropout_regression', 0.0)
        self.regression_head1 = self._build_regression_head(in_features_regressor,
                                                            model_config['regression_hidden_sizes'],
                                                            dropout_reg)
        if self.use_multi_head:
            self.regression_head2 = self._build_regression_head(in_features_regressor,
                                                                model_config['regression_hidden_sizes'],
                                                                dropout_reg)

    def _build_mlp_backbone(self, input_dim, hidden_sizes, dropout_rate):
        """Helper to build MLP layers."""
        layers = []
        in_features = input_dim
        activation_layer = self.activation_layer()
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation_layer)
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_features = hidden_size
        return nn.Sequential(*layers), in_features  # Return model and output feature size

    def _build_cnn1d_backbone(self):
        """Helper to build 1D CNN backbone."""
        layers = []
        in_channels = 1  # Input is (B, 1, L)
        current_seq_len = self.input_size
        use_bn = self.model_config.get('use_batchnorm', True)
        dropout_shared = self.model_config.get('dropout_shared', 0.0)
        activation_layer = self.activation_layer()

        cnn_channels = self.model_config['cnn_channels']
        cnn_kernels = self.model_config['cnn_kernels']
        cnn_pools = self.model_config['cnn_pools']

        for out_channels, kernel_size, pool_size in zip(cnn_channels, cnn_kernels, cnn_pools):
            # Same padding for Conv1d: padding = kernel_size // 2 works for odd kernels
            # For even kernels, PyTorch padding='same' is needed (or manual asymmetric padding)
            padding = kernel_size // 2 if kernel_size % 2 != 0 else 'same'  # Be careful with 'same' in older pytorch

            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(activation_layer)
            if dropout_shared > 0:
                layers.append(nn.Dropout(p=dropout_shared))
            if pool_size > 1:
                layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
                # Calculate sequence length reduction (approximate for 'same' padding)
                # This calculation might be slightly off with 'same' padding, but good enough for pos encoding max_len
                current_seq_len = math.ceil(current_seq_len / pool_size)

            in_channels = out_channels

        return nn.Sequential(*layers), in_channels, int(current_seq_len)  # Return model, final_channels, final_seq_len

    def _build_regression_head(self, input_features, hidden_sizes, dropout_rate):
        """Helper to build the regression MLP head."""
        layers = []
        in_f = input_features
        activation_layer = self.activation_layer()
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_f, hidden_size))
            layers.append(activation_layer)
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_f = hidden_size
        # Final layer to output a single value
        layers.append(nn.Linear(in_f, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.
        Input x: (batch_size, input_size) - assuming features are flattened per pixel.
        """
        batch_size = x.shape[0]
        shared_output = None  # This will hold the features fed into the regression head

        # --- Reshape Input and Pass Through Backbone ---
        if self.model_type == 'mlp':
            # Input x is already (B, Features)
            shared_output = self.shared_backbone(x)

        elif self.model_type == 'cnn1d':
            # Reshape to (B, C=1, L=input_size)
            x = x.unsqueeze(1)
            x = self.shared_backbone(x)  # Output: (B, C_final, L_reduced)
            # Apply global pooling
            x = self.pooling_aggregator(x)  # Output: (B, C_final, 1)
            shared_output = x.squeeze(-1)  # Output: (B, C_final)

        elif self.model_type == 'cnn_attnpool':
            # Reshape to (B, C=1, L=input_size)
            x = x.unsqueeze(1)
            x = self.shared_backbone(x)  # Output: (B, C_final, L_reduced)
            # Apply attention pooling
            shared_output = self.pooling_aggregator(x)  # Output: (B, C_final)

        elif self.model_type == 'cnn_selfattn':
            # Reshape to (B, C=1, L=input_size)
            x = x.unsqueeze(1)
            x = self.shared_backbone(x)  # Output: (B, C_final, L_reduced)
            # Permute for MHA: (B, L_reduced, C_final)
            x = x.permute(0, 2, 1)
            # Add positional encoding
            if self.pos_encoder:
                x = self.pos_encoder(x)
            # Apply Multi-Head Self-Attention + Residual + Norm
            attn_output, _ = self.self_attention_components['mha'](x, x, x)  # Q, K, V are the same
            x = x + attn_output  # Residual connection
            x = self.self_attention_components['norm'](x)
            # Aggregate: Mean pooling over sequence length dimension
            shared_output = x.mean(dim=1)  # Output: (B, C_final)

        elif self.model_type == 'transformer':
            # Reshape to (B, C=1, L=input_size) for Conv1d patch embedding
            x = x.unsqueeze(1)
            # 1. Patch Embedding
            x = self.transformer_components['patch_embed'](x)  # Output: (B, d_model, num_patches_raw)
            # Ensure correct sequence length if input wasn't perfectly divisible
            if x.shape[2] != self.num_patches - (1 if self.use_cls_token else 0):
                # This can happen if padding was implicitly added by Conv1d or input size wasn't divisible
                # Option 1: Trim/Pad if needed (simpler)
                target_len = self.num_patches - (1 if self.use_cls_token else 0)
                if x.shape[2] > target_len:
                    x = x[:, :, :target_len]
                # Option 2: Re-evaluate self.num_patches based on actual output (more complex)
                # print(f"Adjusting num_patches based on Conv1d output from {self.num_patches} to {x.shape[2] + (1 if self.use_cls_token else 0)}")
                # self.num_patches = x.shape[2] + (1 if self.use_cls_token else 0)

            # Permute for Transformer Encoder: (B, num_patches_raw, d_model)
            x = x.permute(0, 2, 1)

            if self.use_cls_token and self.cls_token is not None:  # Check if cls_token exists
                # cls_token = self.transformer_components['cls_token'].expand(batch_size, -1, -1) # <-- CHANGE THIS LINE
                cls_token = self.cls_token.expand(batch_size, -1, -1)  # <-- ACCESS DIRECTLY
                x = torch.cat((cls_token, x), dim=1)  # (B, effective_num_patches, d_model)

            # 3. Add Positional Encoding
            if self.pos_encoder:
                x = self.pos_encoder(x)  # Input (B, num_patches, d_model)

            # 4. Transformer Encoder
            x = self.transformer_components['encoder'](x)  # Output (B, num_patches, d_model)
            x = self.transformer_components['norm'](x)  # Optional LayerNorm

            # 5. Aggregate Output
            if self.use_cls_token:
                shared_output = x[:, 0]  # Select the CLS token output (B, d_model)
            else:
                # Mean pooling over the sequence dimension
                shared_output = x.mean(dim=1)  # Output: (B, d_model)

        # --- Final Regression Head(s) ---
        if shared_output is None:
            raise RuntimeError("Shared output was not computed. Check model type and forward logic.")

        regression_output1 = self.regression_head1(shared_output)

        if self.use_multi_head:
            regression_output2 = self.regression_head2(shared_output)
            return regression_output1, regression_output2
        else:
            return regression_output1


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- Helper for Gumbel Noise ---
def sample_gumbel(shape, eps=1e-20):
    """Sample Gumbel(0, 1) noise."""
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


# --- New Module: Spectral Top-K Tokenizer ---
class SpectralTopKTokenizer(nn.Module):
    def __init__(self, num_bands, K, embed_dim, attn_hidden_dim=None, dropout=0.0, initial_tau=1.0):
        """
        Selects Top-K spectral bands using Gumbel-Softmax STE and projects them.

        Args:
            num_bands (int): Number of spectral bands (C).
            K (int): Number of top bands to select.
            embed_dim (int): Output embedding dimension for the transformer.
            attn_hidden_dim (int, optional): Hidden dimension for the attention MLP.
                                           Defaults to num_bands // 2.
            dropout (float): Dropout rate after projection.
            initial_tau (float): Initial temperature for Gumbel-Softmax. Can be annealed.
        """
        super().__init__()
        if K > num_bands:
            raise ValueError("K cannot be greater than num_bands")
        self.num_bands = num_bands
        self.K = K
        self.embed_dim = embed_dim
        self.attn_hidden_dim = attn_hidden_dim if attn_hidden_dim is not None else max(1, num_bands // 2)
        self.tau = initial_tau  # Temperature

        # MLP to generate attention scores
        self.attention_mlp = nn.Sequential(
            nn.LayerNorm(num_bands),
            nn.Linear(num_bands, self.attn_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.attn_hidden_dim, num_bands)
        )

        # Projection layer for the selected K bands (used in forward pass)
        self.projection_K = nn.Linear(K, embed_dim)

        # Projection layer for all C bands (used in STE backward pass)
        self.projection_C = nn.Linear(num_bands, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)  # Normalize final tokens

    def forward(self, x):
        """
        Forward pass with STE for training, hard selection for inference.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Tokens of shape (B, H * W, embed_dim).
            torch.Tensor: Indices of selected bands (B, H * W, K). (Optional)
        """
        B, C, H, W = x.shape
        if C != self.num_bands:
            raise ValueError(f"Input channel dim ({C}) doesn't match num_bands ({self.num_bands})")

        # Reshape for pixel-wise processing: (B, C, H, W) -> (B * H * W, C)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C)  # Use -1 for B*H*W

        # 1. Calculate attention scores
        attn_scores = self.attention_mlp(x_reshaped)  # (B*H*W, C)

        # --- Training Mode (with Gumbel noise and STE) ---
        if self.training:
            # 2. Add Gumbel noise
            gumbel = sample_gumbel(attn_scores.shape).to(attn_scores.device)
            noisy_scores = attn_scores + gumbel

            # 3. Top-K Selection (Forward - Hard) based on noisy scores
            _, topk_indices = torch.topk(noisy_scores, self.K, dim=1)  # (B*H*W, K)

            # 4. Gather Bands (Forward - Hard)
            # Ensure indices are correct shape for gather: (B*H*W, K)
            # Gather requires indices to match the dimension size otherwise broadcasting happens.
            # We need to expand x_reshaped or indices if necessary, but here they match B*H*W.
            # We gather along dim 1 (the Channel dim).
            # Indices need to be LongTensor
            selected_bands_hard = torch.gather(x_reshaped, 1, topk_indices.long())  # (B*H*W, K)

            # 5. Projection K (Forward - Hard)
            token_hard = self.projection_K(selected_bands_hard)  # (B*H*W, embed_dim)

            # --- Differentiable Proxy (Backward via STE) ---
            # 6a. Calculate soft probabilities (Gumbel-Softmax)
            y_soft = F.softmax(noisy_scores / self.tau, dim=1)  # (B*H*W, C)

            # 6b. Weighted Spectrum (Soft)
            weighted_spectrum_soft = x_reshaped * y_soft  # (B*H*W, C)

            # 6c. Projection C (Soft)
            token_soft = self.projection_C(weighted_spectrum_soft)  # (B*H*W, embed_dim)

            # 7. Straight-Through Estimator
            # Gradient flows back through token_soft, forward uses token_hard
            tokens = token_hard + (token_soft - token_soft.detach())

        # --- Evaluation Mode (Hard selection, no noise/STE) ---
        else:
            # 3. Top-K Selection (Hard) based on original scores
            _, topk_indices = torch.topk(attn_scores, self.K, dim=1)  # (B*H*W, K)

            # 4. Gather Bands (Hard)
            selected_bands_hard = torch.gather(x_reshaped, 1, topk_indices.long())  # (B*H*W, K)

            # 5. Projection K (Hard)
            tokens = self.projection_K(selected_bands_hard)  # (B*H*W, embed_dim)

        # Apply normalization and dropout
        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)

        # Reshape back to sequence format for Transformer: (B, H * W, embed_dim)
        tokens = tokens.reshape(B, H * W, self.embed_dim)

        # Reshape indices for potential inspection: (B, H * W, K)
        indices_output = topk_indices.reshape(B, H * W, self.K)

        return tokens, indices_output  # Return tokens and selected indices


# --- New Module: Spectral Attention Tokenizer ---
class SpectralAttentionTokenizer(nn.Module):
    def __init__(self, num_bands, embed_dim, attn_hidden_dim=None, dropout=0.0):
        """
        Applies attention across the spectral dimension for each pixel and projects
        the re-weighted spectrum to the embedding dimension.

        Args:
            num_bands (int): Number of spectral bands (C).
            embed_dim (int): Output embedding dimension for the transformer.
            attn_hidden_dim (int, optional): Hidden dimension for the attention MLP.
                                           Defaults to num_bands // 2 or 1 if num_bands=1.
            dropout (float): Dropout rate after projection.
        """
        super().__init__()
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        if attn_hidden_dim is None:
            attn_hidden_dim = max(1, num_bands // 2)

        # Simple MLP to generate attention scores for each band
        self.attention_mlp = nn.Sequential(
            nn.LayerNorm(num_bands),  # Normalize features before attention
            nn.Linear(num_bands, attn_hidden_dim),
            nn.Tanh(),  # Tanh is common for attention score predictors
            nn.Linear(attn_hidden_dim, num_bands)
        )

        # Linear layer to project the re-weighted spectrum to embed_dim
        self.projection = nn.Linear(num_bands, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)  # Optional: Normalize tokens

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Tokens of shape (B, H * W, embed_dim).
            torch.Tensor: Attention weights of shape (B, H * W, C). (Optional, for inspection)
        """
        B, C, H, W = x.shape
        if C != self.num_bands:
            raise ValueError(f"Input channel dim ({C}) doesn't match num_bands ({self.num_bands})")

        # Reshape for pixel-wise processing: (B, C, H, W) -> (B, H, W, C) -> (B * H * W, C)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Calculate attention scores
        attn_scores = self.attention_mlp(x_reshaped)  # (B * H * W, C)

        # Get attention weights using softmax
        attn_weights = F.softmax(attn_scores, dim=1)  # (B * H * W, C)

        # Re-weight the original spectral features
        reweighted_spectrum = x_reshaped * attn_weights  # (B * H * W, C)

        # Project to embedding dimension
        tokens = self.projection(reweighted_spectrum)  # (B * H * W, embed_dim)
        tokens = self.norm(tokens)  # Apply layer norm
        tokens = self.dropout(tokens)

        # Reshape back to sequence format for Transformer: (B, H * W, embed_dim)
        tokens = tokens.reshape(B, H * W, self.embed_dim)

        # Reshape attention weights for potential inspection: (B, H * W, C)
        attn_weights_output = attn_weights.reshape(B, H * W, C)

        return tokens, attn_weights_output  # Return tokens and weights


# --- Modify the Main Predictor Class ---
class SpectralPredictorCostum(nn.Module):
    def __init__(self, model_config):
        # --- Add 'spectral_topk_transformer' to handled types ---
        super().__init__()
        self.model_config = model_config
        self.model_type = model_config['model_type']
        # ... (other initializations as before) ...
        self.num_bands = model_config['num_bands']
        self.patch_height = model_config['patch_height']
        self.patch_width = model_config['patch_width']
        self.use_pos_embed = model_config.get('use_pos_embed', True)
        self.use_cls_token = model_config.get('use_cls_token', False)
        self.use_multi_head = model_config.get('use_multi_head', False)
        self.activation_type = model_config.get('activation_type', 'relu').lower()
        if self.activation_type not in ACTIVATION_MAP:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")
        self.activation_layer = ACTIVATION_MAP[self.activation_type]

        self.cnn_backbone = None
        self.patch_embed = None
        self.spectral_tokenizer = None  # For soft attention
        self.spectral_topk_tokenizer = None  # For top-k attention
        self.transformer_encoder = None
        self.pos_embed = None
        self.shared_layers = None
        self.regression_head1 = None
        self.regression_head2 = None

        in_features_regressor = 0
        current_h, current_w, current_c = self.patch_height, self.patch_width, self.num_bands

        # --- Build Model Components ---
        # 2. --- Patch Embedding (for transformer types) ---
        if self.model_type in ['transformer', 'cnn_transformer']:
            self.t_patch_size = model_config['transformer_patch_size']
            if current_h % self.t_patch_size != 0 or current_w % self.t_patch_size != 0:
                for new_patch_size in range(self.t_patch_size, 0, -1):
                    if current_h % new_patch_size == 0 and current_w % new_patch_size == 0:
                        self.t_patch_size = new_patch_size
                        print(
                            f"Adjusted patch size to {self.t_patch_size} for input dimensions ({current_h}, {current_w})")
                        break

            # self.num_patches = (current_h // self.t_patch_size) * (current_w // self.t_patch_size)
            num_patches_h = math.floor((current_h - self.t_patch_size) / model_config['transformer_stride']) + 1
            num_patches_w = math.floor((current_w - self.t_patch_size) / model_config['transformer_stride']) + 1
            self.num_patches = num_patches_h * num_patches_w
            patch_dim = current_c * self.t_patch_size * self.t_patch_size  # Flattened size of one sub-patch

            self.t_embed_dim = model_config['transformer_embed_dim']
            self.t_nheads = model_config['transformer_heads']
            if self.t_embed_dim % self.t_nheads != 0:
                print(f"Embedding dimension {self.t_embed_dim} must be divisible by number of heads {self.t_nheads}.")
                self.t_nheads = self.t_embed_dim // 2  # Adjust to be divisible by 2
            # Efficient patching and projection using a Conv2d
            self.patch_embed = nn.Conv2d(current_c, self.t_embed_dim,
                                         kernel_size=self.t_patch_size, stride=model_config['transformer_stride'])

            # --- Positional Embedding & CLS Token ---
            if self.use_cls_token and self.use_pos_embed:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.t_embed_dim))
                # Add 1 to sequence length for CLS token
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.t_embed_dim))
                nn.init.trunc_normal_(self.pos_embed, std=.02)  # Initialize positional embedding
                nn.init.trunc_normal_(self.cls_token, std=.02)
            elif self.use_pos_embed:
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.t_embed_dim))
                nn.init.trunc_normal_(self.pos_embed, std=.02)

            # --- Transformer Encoder ---
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.t_embed_dim,
                nhead=self.t_nheads,
                dim_feedforward=model_config['transformer_mlp_dim'],
                dropout=model_config['transformer_dropout'],
                activation=self.activation_type,  # or 'gelu'
                batch_first=True  # Important: Input shape (B, Seq, Emb)
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=model_config['transformer_depth']
            )
            in_features_regressor = self.t_embed_dim  # Input to regressor is the transformer output dim

        if self.model_type == 'spectral_transformer':  # Keep the soft version
            # --- Spectral Attention Transformer (Soft) ---
            self.t_embed_dim = model_config['transformer_embed_dim']
            self.t_nheads = model_config['transformer_heads']
            # ... (head divisibility check) ...
            if self.t_embed_dim % self.t_nheads != 0:
                print(
                    f"Warning: Embedding dim {self.t_embed_dim} not divisible by heads {self.t_nheads}. Adjusting heads.")
                for h in range(self.t_nheads, 0, -1):
                    if self.t_embed_dim % h == 0:
                        self.t_nheads = h
                        break
                print(f"Adjusted heads to {self.t_nheads}")

            self.spectral_tokenizer = SpectralAttentionTokenizer(  # Using the soft version class
                num_bands=self.num_bands,
                embed_dim=self.t_embed_dim,
                attn_hidden_dim=model_config.get('spectral_attn_hidden_dim', None),
                dropout=model_config.get('transformer_dropout', 0.0)
            )
            self.num_patches = self.patch_height * self.patch_width
            if self.use_pos_embed:
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.t_embed_dim))
                nn.init.trunc_normal_(self.pos_embed, std=.02)
            encoder_layer = nn.TransformerEncoderLayer(...)  # As before
            self.transformer_encoder = nn.TransformerEncoder(...)  # As before
            in_features_regressor = self.t_embed_dim

        elif self.model_type == 'spectral_topk_transformer':
            # --- NEW: Spectral Top-K Transformer ---
            self.t_embed_dim = model_config['transformer_embed_dim']
            self.t_nheads = model_config['transformer_heads']
            # ... (head divisibility check) ...
            if self.t_embed_dim % self.t_nheads != 0:
                print(
                    f"Warning: Embedding dim {self.t_embed_dim} not divisible by heads {self.t_nheads}. Adjusting heads.")
                for h in range(self.t_nheads, 0, -1):
                    if self.t_embed_dim % h == 0:
                        self.t_nheads = h
                        break
                print(f"Adjusted heads to {self.t_nheads}")

            self.K = model_config['spectral_topk_k']  # Need K in config
            self.initial_tau = model_config.get('spectral_gumbel_tau', 1.0)  # Optional tau in config

            self.spectral_topk_tokenizer = SpectralTopKTokenizer(
                num_bands=self.num_bands,
                K=self.K,
                embed_dim=self.t_embed_dim,
                attn_hidden_dim=model_config.get('spectral_attn_hidden_dim', None),
                dropout=model_config.get('transformer_dropout', 0.0),
                initial_tau=self.initial_tau
            )
            self.num_patches = self.patch_height * self.patch_width
            if self.use_pos_embed:
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.t_embed_dim))
                nn.init.trunc_normal_(self.pos_embed, std=.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.t_embed_dim, nhead=self.t_nheads,
                dim_feedforward=model_config['transformer_mlp_dim'],
                dropout=model_config['transformer_dropout'],
                activation=self.activation_type, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=model_config['transformer_depth']
            )
            in_features_regressor = self.t_embed_dim

        # ... (Rest of the __init__ for cnn_transformer, transformer, mlp, cnn1d, cnn2d) ...
        # ... (Regression head definitions remain the same) ...
        # 4. --- Regression Head 1 ---
        reg_layers1 = []
        in_features1 = in_features_regressor
        activation_layer_reg1 = self.activation_layer()

        for hidden_size in model_config['regression_hidden_sizes']:
            reg_layers1.append(nn.Dropout(p=model_config['dropout_regression']))
            reg_layers1.append(nn.Linear(in_features1, hidden_size))
            reg_layers1.append(activation_layer_reg1)
            in_features1 = hidden_size
        reg_layers1.append(nn.Dropout(p=model_config['dropout_regression']))
        reg_layers1.append(nn.Linear(in_features1, 1))
        self.regression_head1 = nn.Sequential(*reg_layers1)

        # 5. --- Regression Head 2 ---
        reg_layers2 = []
        in_features2 = in_features_regressor
        activation_layer_reg2 = self.activation_layer()

        for hidden_size in model_config['regression_hidden_sizes']:
            reg_layers2.append(nn.Dropout(p=model_config['dropout_regression']))
            reg_layers2.append(nn.Linear(in_features2, hidden_size))
            reg_layers2.append(activation_layer_reg2)
            in_features2 = hidden_size
        reg_layers2.append(nn.Dropout(p=model_config['dropout_regression']))
        reg_layers2.append(nn.Linear(in_features2, 1))
        self.regression_head2 = nn.Sequential(*reg_layers2)

    # _build_cnn_backbone and _calculate_cnn_output_dim remain the same

    # --- Add temperature annealing logic (Optional) ---
    def update_tau(self, current_epoch, total_epochs, final_tau=0.1):
        """Anneal Gumbel-Softmax temperature."""
        if hasattr(self, 'spectral_topk_tokenizer') and self.spectral_topk_tokenizer is not None:
            # Example: Cosine annealing
            # new_tau = final_tau + (self.initial_tau - final_tau) * \
            #           (1 + math.cos(math.pi * current_epoch / total_epochs)) / 2
            # Example: Exponential decay
            decay_rate = (final_tau / self.initial_tau) ** (1.0 / max(1, total_epochs - 1))
            new_tau = self.initial_tau * (decay_rate ** current_epoch)
            new_tau = max(final_tau, new_tau)  # Ensure it doesn't go below final_tau
            self.spectral_topk_tokenizer.tau = new_tau
            # print(f"Epoch {current_epoch}: Updated Gumbel tau to {self.spectral_topk_tokenizer.tau:.4f}")

    def forward(self, x):
        """
                Forward pass.
                Input x: (batch_size, patch_height, patch_width, num_bands)
                """
        # --- Input Permutation/Reshaping ---
        # PyTorch CNNs/Transformers usually expect Channels first
        # Target shape: (B, C, H, W) for Conv2D/Transformer patching
        # Target shape: (B, C, L) for Conv1D where L=H*W
        # # Target shape: (B, Features) for MLP
        # if self.model_type in ['cnn2d', 'transformer', 'cnn_transformer']:
        #      # (B, H, W, C) -> (B, C, H, W)
        #      x = x.permute(0, 3, 1, 2)
        if self.model_type == 'cnn1d':
            # (B, H, W, C) -> (B, C, H*W)
            # x = x.permute(0, 3, 1, 2)
            x = x.reshape(x.size(0), self.num_bands, -1)
        elif self.model_type == 'mlp':
            # (B, H, W, C) -> (B, H*W*C)
            x = x.view(x.size(0), -1)

        # --- Model Body ---
        shared_output = None
        selected_indices = None  # To store indices if needed
        if self.model_type == 'transformer':
            # 1. Patch Embedding (applied directly to input)
            x = self.patch_embed(x)  # Output: (B, embed_dim, num_patches_h, num_patches_w)
            x = x.flatten(2)  # Output: (B, embed_dim, num_patches)
            x = x.transpose(1, 2)  # Output: (B, num_patches, embed_dim)
            # 2. Add CLS token and Positional Embedding
            if self.use_cls_token and self.use_pos_embed:
                batch_size = x.shape[0]
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.pos_embed
            elif self.use_pos_embed:
                x = x + self.pos_embed[:, :self.num_patches, :]
            # 3. Transformer Encoder
            x = self.transformer_encoder(x)
            # 4. Aggregate output
            if self.use_cls_token:
                shared_output = x[:, 0]
            else:
                shared_output = x.mean(dim=1)
        # --- Handle the new model type ---
        elif self.model_type == 'spectral_transformer':  # Soft attention
            x, _ = self.spectral_tokenizer(x)  # Output: (B, H*W, embed_dim), weights ignored here
            if self.use_pos_embed:
                if self.pos_embed is None or self.pos_embed.shape[1] != x.shape[1] or self.pos_embed.shape[2] != \
                        x.shape[2]:
                    raise ValueError("Positional embedding shape mismatch.")
                x = x + self.pos_embed
            x = self.transformer_encoder(x)
            shared_output = x.mean(dim=1)

        elif self.model_type == 'spectral_topk_transformer':  # Top-K attention
            # 1. Get pixel tokens using Top-K selection
            x, selected_indices = self.spectral_topk_tokenizer(x)  # Output: (B, H*W, embed_dim), (B, H*W, K)
            # Note: selected_indices can be returned or stored

            # 2. Add Positional Embedding
            if self.use_pos_embed:
                if self.pos_embed is None or self.pos_embed.shape[1] != x.shape[1] or self.pos_embed.shape[2] != \
                        x.shape[2]:
                    raise ValueError("Positional embedding shape mismatch.")
                x = x + self.pos_embed

            # 3. Transformer Encoder
            x = self.transformer_encoder(x)  # Output: (B, H*W, embed_dim)

            # 4. Aggregate output - Mean pooling
            shared_output = x.mean(dim=1)  # (B, embed_dim)

        # ... (Rest of the forward pass for cnn_transformer, transformer, mlp, cnn1d, cnn2d) ...

        # --- Final Regression Heads ---
        if shared_output is None:
            raise RuntimeError("Model forward pass did not produce an output for the regression head.")

        regression_output1 = self.regression_head1(shared_output)

        if self.use_multi_head:
            regression_output2 = self.regression_head2(shared_output)
            # Optionally return indices
            # if selected_indices is not None:
            #     return regression_output1, regression_output2, selected_indices
            # else:
            return regression_output1, regression_output2
        else:
            # Optionally return indices
            # if selected_indices is not None:
            #     return regression_output1, selected_indices
            # else:
            return regression_output1



class EncoderAE(nn.Module):
    def __init__(self, config, input_channels, input_height, input_width):
        super().__init__()
        self.config = config
        self.activation_layer = ACTIVATION_MAP[config.get('encoder_activation', 'relu')]()

        # Example: Simple CNN Encoder
        # You can make this much more configurable like your SpectralPredictorWithTransformer
        self.conv1 = nn.Conv2d(input_channels, config.get('encoder_channels', [128, 64])[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(config.get('encoder_channels', [128, 64])[0])
        self.conv2 = nn.Conv2d(config.get('encoder_channels', [128, 64])[0], config.get('encoder_channels', [128, 64])[1],
                               kernel_size=3,
                               stride=1, padding=1)  # Downsample
        self.bn2 = nn.BatchNorm2d(config.get('encoder_channels', [128, 64])[1])
        self.conv3 = nn.Conv2d(config.get('encoder_channels', [128, 64])[1], config.get('encoder_channels', [128, 64,32])[2],
                               kernel_size=3, stride=1, padding=1)  # Downsample
        self.bn3 = nn.BatchNorm2d(config.get('encoder_channels', [128, 64,32])[2])

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            dummy_out = self.bn3(self.activation_layer(self.conv3(
                self.bn2(self.activation_layer(self.conv2(self.bn1(self.activation_layer(self.conv1(dummy_input)))))))))
            self.flattened_size = dummy_out.numel() // dummy_out.shape[0]

        self.fc_mu = nn.Linear(self.flattened_size, config['latent_dim'])
        if config.get('variational', False):  # Optional: Variational AE
            self.fc_logvar = nn.Linear(self.flattened_size, config['latent_dim'])
        self.is_variational = config.get('variational', False)

    def forward(self, x):
        x = self.activation_layer(self.bn1(self.conv1(x)))
        x = self.activation_layer(self.bn2(self.conv2(x)))
        x = self.activation_layer(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        if self.is_variational:
            logvar = self.fc_logvar(x)
            return mu, logvar
        return mu



class DecoderAE(nn.Module):
    def __init__(self, config, output_channels,  # output_channels is the original input_channels to AE
                 encoder_conv_output_shape):
        super().__init__()
        self.config = config
        self.activation_layer = ACTIVATION_MAP[config.get('decoder_activation', 'relu')]()  # Use specific key

        # encoder_conv_output_shape will be (C_deep, original_input_height, original_input_width)
        # C_deep = config.get('channels_deep')[0] from encoder config
        self.encoder_C_deep = encoder_conv_output_shape[0]
        self.encoder_H = encoder_conv_output_shape[1]
        self.encoder_W = encoder_conv_output_shape[2]

        # FC layer to project latent_dim back to the flattened size of the encoder's last conv output
        self.fc_input_size = self.encoder_C_deep * self.encoder_H * self.encoder_W
        self.fc = nn.Linear(config['latent_dim'], self.fc_input_size)

        decoder_tconv_channels = config.get('decoder_channels', [self.config.get('channels', [128, 64])[1],
                                                                 self.config.get('channels', [128, 64])[0]])
        # channels_deep from encoder config: self.config.get('channels_deep', [32])[0]

        self.tconv1 = nn.ConvTranspose2d(
            in_channels=self.encoder_C_deep,  # e.g., 32 from encoder's channels_deep[0]
            out_channels=decoder_tconv_channels[0],  # e.g., 64 (config.get('channels')[1] from encoder)
            kernel_size=3, stride=1, padding=1  # No spatial upsampling
        )
        self.bn1 = nn.BatchNorm2d(decoder_tconv_channels[0])

        self.tconv2 = nn.ConvTranspose2d(
            in_channels=decoder_tconv_channels[0],  # e.g., 64
            out_channels=decoder_tconv_channels[1],  # e.g., 128 (config.get('channels')[0] from encoder)
            kernel_size=3, stride=1, padding=1  # No spatial upsampling
        )
        self.bn2 = nn.BatchNorm2d(decoder_tconv_channels[1])

        self.tconv3 = nn.ConvTranspose2d(
            in_channels=decoder_tconv_channels[1],  # e.g., 128
            out_channels=output_channels,  # Original input_channels to the AE
            kernel_size=3, stride=1, padding=1  # No spatial upsampling
        )

        # Final activation (Sigmoid or Tanh based on reconstruction target range)
        final_act_str = config.get('decoder_final_activation', 'sigmoid').lower()
        if final_act_str == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_act_str == 'tanh':
            self.final_activation = nn.Tanh()
        else:  # Raw output
            self.final_activation = nn.Identity()

    def forward(self, z):  # z is the latent code
        x = self.fc(z)
        # Reshape to match the output of the encoder's last conv layer
        # Shape: (batch_size, self.encoder_C_deep, self.encoder_H, self.encoder_W)
        x = x.view(x.size(0), self.encoder_C_deep, self.encoder_H, self.encoder_W)

        x = self.activation_layer(self.bn1(self.tconv1(x)))
        # After tconv1, x shape: (batch_size, decoder_tconv_channels[0], self.encoder_H, self.encoder_W)

        x = self.activation_layer(self.bn2(self.tconv2(x)))
        # After tconv2, x shape: (batch_size, decoder_tconv_channels[1], self.encoder_H, self.encoder_W)

        x = self.tconv3(x)  # No BN or intermediate activation before final output activation
        # After tconv3, x shape: (batch_size, output_channels, self.encoder_H, self.encoder_W)

        reconstructed_x = self.final_activation(x)
        return reconstructed_x


class EncoderAE_Flexi(nn.Module):
    def __init__(self, config, input_channels, input_height, input_width):  # Python style: __init__
        super().__init__()
        self.config = config
        self.activation_fn = ACTIVATION_MAP[config.get('encoder_activation', 'relu')]()  # Instantiate activation

        encoder_conv_channels = config.get('encoder_channels', [128, 64, 32])  # e.g., [C1, C2, C_deep]
        encoder_kernels = config.get('encoder_kernels', [3, 1, 1])  # e.g., [K1, K2, K_deep]
        if not encoder_conv_channels:  # Handle empty case: direct to FC
            self.conv_layers = nn.ModuleList()
            self.bn_layers = nn.ModuleList()
            self.final_conv_out_channels = input_channels  # No convs, so input channels go to FC
            # In this case, flattened_size would be input_channels * input_height * input_width
            # The fc_mu and fc_logvar would take this directly.
        else:
            self.conv_layers = nn.ModuleList()
            self.bn_layers = nn.ModuleList()

            current_channels = input_channels
            for out_channels, kernel_size in zip(encoder_conv_channels, encoder_kernels):
                padding = kernel_size // 2
                self.conv_layers.append(
                    nn.Conv2d(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,  # Make kernel size configurable
                        stride=1,  # Make stride configurable
                        padding= padding  # Make padding configurable
                    )
                )
                self.bn_layers.append(nn.BatchNorm2d(out_channels))
                current_channels = out_channels
            self.final_conv_out_channels = current_channels  # Channels after the last conv layer

        # Calculate flattened size dynamically based on the constructed conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_output_for_flattening = dummy_input
            if self.conv_layers:  # Only pass through convs if they exist
                for i in range(len(self.conv_layers)):
                    conv_output_for_flattening = self.conv_layers[i](conv_output_for_flattening)
                    conv_output_for_flattening = self.bn_layers[i](conv_output_for_flattening)
                    conv_output_for_flattening = self.activation_fn(conv_output_for_flattening)

            # Store the shape before flattening for the decoder
            self.shape_before_flatten = conv_output_for_flattening.shape[1:]  # (C, H, W)
            self.flattened_size = conv_output_for_flattening.numel() // conv_output_for_flattening.shape[0]

        self.fc_mu = nn.Linear(self.flattened_size, config['latent_dim'])
        self.is_variational = config.get('variational', False)
        if self.is_variational:
            self.fc_logvar = nn.Linear(self.flattened_size, config['latent_dim'])

    # In EncoderAE_Flexi
    def forward(self, x):
        x_conv_out = x
        # Convolutional part
        for i in range(len(self.conv_layers)):
            x_conv_out = self.conv_layers[i](x_conv_out)
            x_conv_out = self.bn_layers[i](x_conv_out)
            x_conv_out = self.activation_fn(x_conv_out)

        # x_conv_out is now the (B, C, H, W) feature map

        # Flatten for FC layers to get mu and logvar (the traditional latent code 'z')
        x_flattened = x_conv_out.view(x_conv_out.size(0), -1)
        mu = self.fc_mu(x_flattened)

        if self.is_variational:
            logvar = self.fc_logvar(x_flattened)
            # Return mu, logvar for VAE reparameterization, AND the conv features
            return mu, logvar, x_conv_out
        # Return mu (as 'z') AND the conv features
        return mu, x_conv_out


class DecoderAE_Flexi(nn.Module):
    def __init__(self, config,
                 output_channels,  # Original input channels to AE
                 # Shape from encoder's last conv output (or input shape if no encoder convs)
                 # Tuple: (C_last_encoder_conv, H_last_encoder_conv, W_last_encoder_conv)
                 encoder_shape_before_flatten,
                 # List of output channels of encoder convs, e.g., [128, 64, 32]
                 # This helps decoder to reverse the channel progression
                 encoder_channel_progression  # e.g., config.get('encoder_channels')
                 ):
        super().__init__()
        self.config = config
        self.activation_fn = ACTIVATION_MAP[config.get('decoder_activation', 'relu')]()  # Instantiate

        self.C_from_encoder = encoder_shape_before_flatten[0]
        self.H_to_reconstruct = encoder_shape_before_flatten[1]
        self.W_to_reconstruct = encoder_shape_before_flatten[2]

        # FC layer to project latent_dim back to the flattened size of the encoder's output before FC
        self.fc_output_size = self.C_from_encoder * self.H_to_reconstruct * self.W_to_reconstruct
        self.fc = nn.Linear(config['latent_dim'], self.fc_output_size)

        # Transposed Convolutional Layers
        self.tconv_layers = nn.ModuleList()
        self.tbn_layers = nn.ModuleList()

        # Decoder channel progression should be the reverse of encoder's, ending at original input_channels
        # Example:
        # Encoder channels: [E1, E2, E3_deep (self.C_from_encoder)]
        # Decoder needs to go from E3_deep -> E2 -> E1 -> output_channels (original input_channels)
        # So, decoder_conv_out_channels will be [E2, E1, output_channels]

        # Use config.get('decoder_channels', default_list) for explicit control,
        # or derive from encoder_channel_progression.
        # For simplicity, let's derive it here, but explicit config is often better for tuning.
        decoder_kernels = config.get('decoder_kernels', [3, 1, 1])[::-1]
        decoder_target_channels = []
        if encoder_channel_progression:  # If encoder had conv layers
            # Target channels for decoder's tconv outputs, in reverse order of encoder's outputs,
            # then finally the AE's original input channel count.
            # If encoder_channels was [C1, C2, C3_deep], we want decoder to output C2, then C1, then final_output_C
            if len(encoder_channel_progression) > 1:
                decoder_target_channels.extend(list(reversed(encoder_channel_progression[:-1])))  # All but the last
            decoder_target_channels.append(output_channels)  # The final output channel count for AE
        else:  # No convs in encoder, so no tconvs needed in decoder beyond the FC reshape
            pass  # self.tconv_layers will remain empty

        current_channels = self.C_from_encoder  # Start with channels from encoder's last conv output

        if not decoder_target_channels and self.C_from_encoder != output_channels:
            # Edge case: No encoder convs, but input channels to AE is different from latent projection channel count.
            # This typically means we need one tconv layer to match channels if H/W are same.
            # Or, if C_from_encoder (which is input_channels if no enc convs) == output_channels, no tconvs needed.
            if current_channels != output_channels:
                # This scenario is a bit unusual if there were no encoder convs.
                # It implies the fc layer projected to a different channel depth than the target.
                # We might need a 1x1 conv (or a tconv with kernel 1) if H/W are to be preserved.
                # For now, if no encoder_channel_progression, we assume no tconvs are built by this loop.
                # The final_reshape_conv (see below) might handle this channel adjustment.
                pass
        for i, (target_out_channels, kernel_size) in enumerate(zip(decoder_target_channels, decoder_kernels)):
            padding = kernel_size // 2
            self.tconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=current_channels,
                    out_channels=target_out_channels,
                    kernel_size=kernel_size,  # Make configurable
                    stride=1,  # Make configurable
                    padding=padding,  # Make configurable
                    # output_padding is only for stride > 1, usually 0 or stride-kernel_size%stride
                    output_padding=config.get('decoder_output_padding', 0) if config.get('decoder_stride', 1) > 1 else 0
                )
            )
            # No BatchNorm for the very last ConvTranspose if it directly leads to final_activation
            if i < len(decoder_target_channels) - 1:  # Don't add BN after the layer that outputs `output_channels`
                self.tbn_layers.append(nn.BatchNorm2d(target_out_channels))
            elif len(
                    decoder_target_channels) == 1 and current_channels != output_channels:  # Special case: only one tconv layer
                self.tbn_layers.append(nn.BatchNorm2d(target_out_channels))

            current_channels = target_out_channels

        # If after all tconvs, the channel count is not `output_channels` (e.g. no tconvs were made)
        # but C_from_encoder is different, we might need a final 1x1 conv to adjust channels.
        # This is usually handled if decoder_target_channels correctly includes output_channels as the last element.
        self.final_reshape_conv = None
        if not self.tconv_layers and current_channels != output_channels:
            # This case means: Encoder had no convs (or shape_before_flatten is just input shape).
            # FC output (after reshape) has `current_channels`. If this isn't the target `output_channels`,
            # add a 1x1 conv to adjust. This helps if latent_dim projects to something other than `output_channels`.
            print(f"Decoder: Adding a 1x1 conv to map from {current_channels} to {output_channels} channels.")
            self.final_reshape_conv = nn.Conv2d(current_channels, output_channels, kernel_size=1)

        # Final activation
        final_act_str = config.get('decoder_final_activation', 'sigmoid').lower()
        if final_act_str == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_act_str == 'tanh':
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.C_from_encoder, self.H_to_reconstruct, self.W_to_reconstruct)

        if self.final_reshape_conv and not self.tconv_layers:  # Apply if no tconvs and channel mismatch
            x = self.final_reshape_conv(x)

        for i in range(len(self.tconv_layers)):
            x = self.tconv_layers[i](x)
            # Apply BN and Activation for all but potentially the last layer before final_activation
            if i < len(self.tbn_layers):  # Check if a BN layer exists for this conv layer
                x = self.tbn_layers[i](x)
                x = self.activation_fn(x)
            elif i == len(
                    self.tconv_layers) - 1:  # Last tconv layer, activation might be skipped before final_activation
                # If you want activation even before final_activation on the last tconv:
                # x = self.activation_fn(x)
                pass

        reconstructed_x = self.final_activation(x)
        return reconstructed_x



class EncoderAE_3D(nn.Module):
    def __init__(self, config):  # input_size is tuple (H, W) or (D, H, W)
        super().__init__()
        self.config = config
        self.conv_dim = config.get('conv_dim', 2)
        self.use_encoder_fc = config.get('use_encoder_fc', True)
        self.is_variational = config.get('variational', False)  # Original VAE intention
        self.instanceNorm = config.get('encoder_instanceNorm', False)
        input_size = [8,8]
        encoder_conv_channels = config.get('encoder_channels', [128, 64, 32])[:]
        kernel_sizes = config.get('encoder_kernels', [3, 1, 1])[:len(encoder_conv_channels)]  # e.g., [K1, K2, K_deep]
        if self.conv_dim == 3: kernel_sizes.append(1)
        strides = config.get('encoder_strides', [1, 1, 1])[:len(encoder_conv_channels)]
        if self.conv_dim == 3: strides.append(1)
        D3_kernel = config.get('encoder_3D_kernel', 3)
        if self.conv_dim == 2: D3_kernel = None
        if self.conv_dim == 3: encoder_conv_channels.append(1)


        if not self.use_encoder_fc and self.is_variational:
            print("Warning: config.variational is True but use_encoder_fc is False. "
                  "Encoder will output feature maps, not mu/logvar for standard VAE. "
                  "Internally treating as non-variational for FC-related parts.")
            self.is_variational = False  # Override: No FC means no standard mu/logvar from FCs

        self.activation_fn = ACTIVATION_MAP[config.get('encoder_activation', 'relu')]()

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        if self.conv_dim == 3: input_channels = 1
        else: input_channels = 224
        current_channels = input_channels
        ConvNd = nn.Conv2d
        if not encoder_conv_channels:
            self.final_conv_out_channels = input_channels
        else:
            if self.conv_dim == 3: ConvNd = nn.Conv3d
            BatchNormNd = nn.BatchNorm2d if not self.instanceNorm else nn.InstanceNorm2d
            if self.conv_dim == 3:
                BatchNormNd = nn.BatchNorm3d
                if self.instanceNorm:
                    BatchNormNd = nn.InstanceNorm3d

            for i, out_channels in enumerate(encoder_conv_channels):
                kernel_size = [D3_kernel] if self.conv_dim == 3 else []
                kernel_size.extend([kernel_sizes[i],kernel_sizes[i]])
                padding = tuple(k // 2 for k in kernel_size)
                stride = [strides[i]] if self.conv_dim == 3 else []
                stride.extend([1,1])
                self.conv_layers.append(
                    ConvNd(in_channels=current_channels, out_channels=out_channels,
                           kernel_size=tuple(kernel_size), stride=tuple(stride), padding=tuple(padding))
                )
                self.bn_layers.append(BatchNormNd(out_channels))
                current_channels = out_channels
            self.final_conv_out_channels = current_channels

        with torch.no_grad():
            if self.conv_dim == 2:
                dummy_input_shape = (1, 224, input_size[0], input_size[1])
            else:  # conv_dim == 3
                dummy_input_shape = (1, 1, 224, input_size[0], input_size[1])

            dummy_input = torch.zeros(*dummy_input_shape)
            conv_output_for_flattening = dummy_input
            if self.conv_layers:
                for i in range(len(self.conv_layers)):
                    conv_output_for_flattening = self.conv_layers[i](conv_output_for_flattening)
                    conv_output_for_flattening = self.bn_layers[i](conv_output_for_flattening)
                    conv_output_for_flattening = self.activation_fn(conv_output_for_flattening)

            self.shape_before_flatten = conv_output_for_flattening.shape[1:]  # (C, H, W) or (C, D, H, W)
            self.flattened_size = conv_output_for_flattening.numel() // conv_output_for_flattening.shape[0]

        if self.use_encoder_fc:
            if 'latent_dim' not in config:
                raise ValueError("'latent_dim' must be in config if use_encoder_fc is True.")
            self.fc_mu = nn.Linear(self.flattened_size, self.flattened_size)
            if self.is_variational:  # Use the (potentially overridden) self.is_variational
                self.fc_logvar = nn.Linear(self.flattened_size, self.flattened_size)
            else:
                self.fc_logvar = None
        else:
            self.fc_mu = None
            self.fc_logvar = None

    def forward(self, x):
        x_conv_out = x
        if self.conv_dim == 3: x_conv_out = x_conv_out.unsqueeze(1)
        for i in range(len(self.conv_layers)):
            x_conv_out = self.conv_layers[i](x_conv_out)
            x_conv_out = self.bn_layers[i](x_conv_out)
            x_conv_out = self.activation_fn(x_conv_out)
        if self.conv_dim == 3: x_conv_out = x_conv_out.squeeze(1)
        if self.use_encoder_fc:
            x_flattened = x_conv_out.view(x_conv_out.size(0), -1)
            mu = self.fc_mu(x_flattened)
            if self.is_variational and self.fc_logvar is not None:
                logvar = self.fc_logvar(x_flattened)
                return mu, logvar, x_conv_out
            return mu, x_conv_out
        else:  # Not using FC layers
            return x_conv_out, x_conv_out


# class DecoderAE_3D(nn.Module):
#     def __init__(self, config,
#                  encoder_shape_before_flatten,  # (C, D, H, W) or (C, H, W) from encoder
#                  ):
#         super().__init__()
#         self.config = config
#         self.conv_dim = config.get('conv_dim', 2)
#         self.use_decoder_fc = config.get('use_decoder_fc', True)
#         strides = config.get('decoder_strides', [1, 1, 1])[::-1]
#         if self.conv_dim == 3: strides.append(1)
#         D3_kernel = config.get('decoder_3D_kernel', 5)
#         kernel_sizes = config.get('decoder_kernels', [3, 1, 1])  # e.g., [K1, K2, K_deep]
#         if self.conv_dim == 3: kernel_sizes.append(1)
#         encoder_channel_progression = config.get('decoder_channels', [128,64])
#         output_channels = 1 if self.conv_dim == 3 else 224
#         if self.conv_dim == 2: D3_kernel = None
#         if self.conv_dim not in [2, 3]:
#             raise ValueError("conv_dim must be 2 or 3")
#
#         self.activation_fn = ACTIVATION_MAP[config.get('decoder_activation', 'relu')]()
#
#         self.C_from_encoder = encoder_shape_before_flatten[0]
#         if self.conv_dim == 2:
#             if len(encoder_shape_before_flatten) != 3:  # C, H, W
#                 raise ValueError(
#                     f"encoder_shape_before_flatten must be (C, H, W) for conv_dim=2, got {encoder_shape_before_flatten}")
#             self.H_to_reconstruct = encoder_shape_before_flatten[1]
#             self.W_to_reconstruct = encoder_shape_before_flatten[2]
#             self.D_to_reconstruct = None
#         else:  # conv_dim == 3
#             if len(encoder_shape_before_flatten) != 4:  # C, D, H, W
#                 raise ValueError(
#                     f"encoder_shape_before_flatten must be (C, D, H, W) for conv_dim=3, got {encoder_shape_before_flatten}")
#             self.D_to_reconstruct = encoder_shape_before_flatten[1]
#             self.H_to_reconstruct = encoder_shape_before_flatten[2]
#             self.W_to_reconstruct = encoder_shape_before_flatten[3]
#
#         if self.use_decoder_fc:
#             if 'latent_dim' not in config:
#                 raise ValueError("'latent_dim' must be in config if use_decoder_fc is True.")
#             fc_input_dim = config['latent_dim']
#
#             self.fc_output_size = self.C_from_encoder * self.H_to_reconstruct * self.W_to_reconstruct
#             if self.conv_dim == 3 and self.D_to_reconstruct is not None:
#                 self.fc_output_size *= self.D_to_reconstruct
#             self.fc = nn.Linear(fc_input_dim, self.fc_output_size)
#         else:
#             self.fc = None
#             if 'latent_dim' in config:
#                 print(
#                     f"Info: Decoder's 'latent_dim' ({config['latent_dim']}) in config is unused because use_decoder_fc is False.")
#
#         self.tconv_layers = nn.ModuleList()
#         self.tbn_layers = nn.ModuleList()
#
#         # Determine target output channels for each TConv layer
#         decoder_tconv_target_channels = []
#         if self.conv_dim == 3: decoder_tconv_target_channels.append(1)
#         if encoder_channel_progression:  # e.g., [E1, E2, E3_deep], where E3_deep is C_from_encoder
#             if len(encoder_channel_progression) > 1:
#                 # Target TConv output channels: E2, E1 (reversed from encoder, excluding the last one)
#                 decoder_tconv_target_channels.extend(list(reversed(encoder_channel_progression[:-1])))
#             decoder_tconv_target_channels.append(output_channels)  # Final layer targets AE's output_channels
#
#         num_tconv_layers = len(decoder_tconv_target_channels)
#         current_channels = self.C_from_encoder
#
#         if decoder_tconv_target_channels:  # Only build TConv layers if targets are defined
#             TConvNd = nn.ConvTranspose2d if self.conv_dim == 2 else nn.ConvTranspose3d
#             BatchNormNd = nn.BatchNorm2d if self.conv_dim == 2 else nn.BatchNorm3d
#
#             for i, target_out_channels in enumerate(decoder_tconv_target_channels):
#                 kernel_size = [D3_kernel] if self.conv_dim == 3 else []
#                 kernel_size.extend([kernel_sizes[i]]*2)
#                 padding = tuple(k // 2 for k in kernel_size)
#                 stride = [strides[i]] if self.conv_dim == 3 else []
#                 stride.extend([1, 1])
#                 self.tconv_layers.append(
#                     TConvNd(in_channels=current_channels, out_channels=target_out_channels,
#                             kernel_size=kernel_size, stride=stride, padding=padding)
#                 )
#                 # Add BN for intermediate layers. Not for the last TConv if it produces final `output_channels`.
#                 if i < num_tconv_layers - 1:  # Any layer that is NOT the last one
#                     self.tbn_layers.append(BatchNormNd(target_out_channels))
#                 elif target_out_channels != output_channels:  # Last TConv, but it does NOT produce final `output_channels` (unusual)
#                     self.tbn_layers.append(BatchNormNd(target_out_channels))
#                 # Else (last TConv layer AND target_out_channels == output_channels): No BN.
#
#                 current_channels = target_out_channels
#
#
#         final_act_str = config.get('decoder_final_activation', 'sigmoid').lower()
#         if final_act_str not in ACTIVATION_MAP:
#             print(f"Warning: decoder_final_activation '{final_act_str}' not in ACTIVATION_MAP. Using Identity.")
#             final_act_str = 'identity'
#         self.final_activation = ACTIVATION_MAP[final_act_str]()
#
#
#     def forward(self, z):  # z is latent vector (if use_decoder_fc) or feature map
#         x = z
#         if self.use_decoder_fc:
#             if self.fc is None:  # Should not happen if use_decoder_fc is True and init was correct
#                 raise RuntimeError("Decoder: use_decoder_fc is True, but self.fc is None.")
#             x = self.fc(x)  # Output: (Batch, flattened_feature_size)
#
#             # Reshape to feature map
#             if self.conv_dim == 2:
#                 target_shape = (x.size(0), self.C_from_encoder, self.H_to_reconstruct, self.W_to_reconstruct)
#             else:  # conv_dim == 3
#                 target_shape = (x.size(0), self.D_to_reconstruct, self.H_to_reconstruct,
#                                 self.W_to_reconstruct)
#             x = x.view(*target_shape)
#
#         # x is now a feature map: (B, C_from_encoder, [D,], H, W)
#         if self.conv_dim == 3: x = x.unsqueeze(1)
#         for i in range(len(self.tconv_layers)):
#             x = self.tconv_layers[i](x)
#             if i < len(self.tbn_layers):  # True if a BN layer was created for this TConv layer
#                 x = self.tbn_layers[i](x)
#                 x = self.activation_fn(x)  # Apply intermediate activation
#
#         if self.conv_dim == 3: x = x.squeeze(1)
#         reconstructed_x = self.final_activation(x)
#         return reconstructed_x



class DecoderAE_3D(nn.Module):
    def __init__(self, config,
                 encoder_shape_before_flatten,  # (C, D, H, W) or (C, H, W)
                 original_input_spatial_shape=None  # Tuple: (orig_D, orig_H, orig_W) for 3D
                 ):
        super().__init__()
        self.config = config
        self.conv_dim = config.get('conv_dim', 2)
        self.use_decoder_fc = config.get('use_encoder_fc', True)
        self.instanceNorm = config.get('decoder_instanceNorm', False)

        self.original_input_spatial_shape = (224,8,8)
        if self.conv_dim == 3:
            if self.original_input_spatial_shape is None:
                print("Warning: DecoderAE_3D: conv_dim=3 but original_input_spatial_shape not provided. "
                      "Output size matching cannot be guaranteed without it.")
            elif not (isinstance(self.original_input_spatial_shape, tuple) and len(
                    self.original_input_spatial_shape) == 3):
                raise ValueError(
                    "For conv_dim=3, original_input_spatial_shape must be a tuple (orig_D, orig_H, orig_W)."
                )

        decoder_intermediate_output_channels = config.get('decoder_channels', [128, 64])[::-1]
        # Strides for TConv layers (depth-only upsampling based on your original logic)
        encoder_actual_depth_strides = config.get('decoder_strides', [])  # Encoder's depth strides
        self.tconv_depth_strides_for_layers = encoder_actual_depth_strides[:len(decoder_intermediate_output_channels)][::-1]
        # if self.conv_dim == 3: self.tconv_depth_strides_for_layers.append(1)

        if self.conv_dim == 3 : self.tconv_depth_strides_for_layers.insert(0,1)
        D3_kernel_val = config.get('decoder_3D_kernel', 5)  # Depth kernel size
        self.decoder_hw_kernels_config = config.get('decoder_kernels', [3, 3, 3])[:len(decoder_intermediate_output_channels)][::-1]  # HW kernel sizes per layer
        # if self.conv_dim == 3: self.decoder_hw_kernels_config.append(1)
        if self.conv_dim == 3 : self.decoder_hw_kernels_config.insert(0,1)

        # if self.conv_dim == 3: decoder_hw_kernels_config.append(1)
        self.final_ae_output_channels = 1 if self.conv_dim == 3 else 224
        self.activation_fn = ACTIVATION_MAP[config.get('decoder_activation', 'relu')]()
        if self.conv_dim == 2: decoder_intermediate_output_channels.pop()

        self.C_from_encoder = encoder_shape_before_flatten[0]
        if self.conv_dim == 2:
            if len(encoder_shape_before_flatten) != 3:
                raise ValueError(
                    f"encoder_shape_before_flatten must be (C, H, W) for conv_dim=2, got {encoder_shape_before_flatten}")
            self.D_to_reconstruct = None
            self.H_to_reconstruct = encoder_shape_before_flatten[1]
            self.W_to_reconstruct = encoder_shape_before_flatten[2]
        else:  # conv_dim == 3
            if len(encoder_shape_before_flatten) != 4:
                raise ValueError(
                    f"encoder_shape_before_flatten must be (C, D, H, W) for conv_dim=3, got {encoder_shape_before_flatten}")
            self.D_to_reconstruct = encoder_shape_before_flatten[1]
            self.H_to_reconstruct = encoder_shape_before_flatten[2]
            self.W_to_reconstruct = encoder_shape_before_flatten[3]

        if self.use_decoder_fc:
            # fc_input_dim = config['latent_dim']
            fc_output_elements = self.C_from_encoder * self.H_to_reconstruct * self.W_to_reconstruct
            if self.conv_dim == 3 and self.D_to_reconstruct is not None:
                fc_output_elements *= self.D_to_reconstruct
            self.fc = nn.Linear(fc_output_elements, fc_output_elements)
        else:
            self.fc = None

        self.tconv_layers = nn.ModuleList()
        self.tbn_layers = nn.ModuleList()
        print('decoder_channels',decoder_intermediate_output_channels)
        self.tconv_output_channel_targets = []
        if decoder_intermediate_output_channels:
            self.tconv_output_channel_targets.extend(decoder_intermediate_output_channels)
            print('channels after extend', self.tconv_output_channel_targets)
        self.tconv_output_channel_targets.append(self.final_ae_output_channels)
        # if self.conv_dim == 3: self.tconv_output_channel_targets.insert(0,1)
        num_tconv_layers = len(self.tconv_output_channel_targets)
        current_tconv_in_channels = self.C_from_encoder

        # sim_D, sim_H, sim_W are the spatial dimensions of the tensor fed into the TConv layers
        sim_D, sim_H, sim_W = self.D_to_reconstruct, self.H_to_reconstruct, self.W_to_reconstruct
        print(f"Decoder Init: Starting TConv input sim_shape (D,H,W): ({sim_D}, {sim_H}, {sim_W}) "
              f"with C={current_tconv_in_channels}")

        TConvNd = nn.ConvTranspose2d if self.conv_dim == 2 else nn.ConvTranspose3d
        BatchNormNd = nn.BatchNorm2d if not self.instanceNorm else nn.InstanceNorm2d
        if self.conv_dim == 3:
            BatchNormNd = nn.BatchNorm3d
            if self.instanceNorm:
                BatchNormNd = nn.InstanceNorm3d
        # Determine idx_last_upsample_depth for 3D case (layer that targets final depth)
        idx_last_upsample_depth = -1
        if self.conv_dim == 3 and num_tconv_layers > 0:
            temp_stride_d_values = [
                self.tconv_depth_strides_for_layers[i] if i < len(self.tconv_depth_strides_for_layers) else 1
                for i in range(num_tconv_layers)
            ]
            for i_s in range(num_tconv_layers - 1, -1, -1):
                if temp_stride_d_values[i_s] > 1:
                    idx_last_upsample_depth = i_s
                    break
            if idx_last_upsample_depth == -1:  # All strides are 1, or no depth strides provided
                idx_last_upsample_depth = num_tconv_layers - 1  # The last layer is responsible
        # print('strides_', self.tconv_depth_strides_for_layers)
        # print('kernels_', self.decoder_hw_kernels_config)
        # print('tconv_', self.tconv_output_channel_targets)
        if self.conv_dim == 3:
            for i, target_layer_out_channels in enumerate(self.tconv_output_channel_targets):
                kernel_d = D3_kernel_val
                kernel_hw = self.decoder_hw_kernels_config[i] if i < len(
                    self.decoder_hw_kernels_config) else 3  # Default HW kernel

                stride_d = self.tconv_depth_strides_for_layers[i] if i < len(self.tconv_depth_strides_for_layers) else 1
                stride_h, stride_w = 1, 1  # Upsampling only in depth for this example

                # --- Padding and Output Padding Calculation for 3D ---
                # For H, W (strides are 1, kernels usually odd)
                padding_hw = kernel_hw // 2
                op_h, op_w = 0, 0  # For stride=1, output_padding must be 0

                # For D dimension
                default_padding_d = kernel_d // 2
                # Default/heuristic op_d
                default_op_d = stride_d - 1 if stride_d > 1 else 0
                default_op_d = max(0, default_op_d)

                padding_d = default_padding_d
                op_d = default_op_d

                current_target_D = None
                if self.original_input_spatial_shape and self.original_input_spatial_shape[0] is not None:
                    current_target_D = self.original_input_spatial_shape[0]

                if i == idx_last_upsample_depth and current_target_D is not None:
                    # This layer is responsible for hitting the target_D
                    calculated_P, calculated_OP = DecoderAE_3D._calculate_padding_and_op(
                        sim_D, current_target_D, stride_d, kernel_d, default_padding_d
                    )

                    if calculated_P is not None and calculated_OP is not None:
                        is_adjusted = (calculated_P != default_padding_d) or (calculated_OP != default_op_d)
                        padding_d = calculated_P
                        op_d = calculated_OP
                        if is_adjusted:
                            print(
                                f"  Layer {i} (designated depth target): Adjusted P_d={padding_d}, OP_d={op_d} to target D={current_target_D}")
                    else:
                        print(
                            f"  Warning: Layer {i} (designated depth target): Could not calculate P, OP to meet D_target={current_target_D} "
                            f"for D_in={sim_D}, S={stride_d}, K={kernel_d}. Using default P_d={padding_d}, OP_d={op_d}.")
                # --- End Padding and Output Padding Calculation ---

                layer_kernel = (kernel_d, kernel_hw, kernel_hw)
                layer_stride = (stride_d, stride_h, stride_w)
                layer_padding = (padding_d, padding_hw, padding_hw)
                layer_output_padding = (op_d, op_h, op_w)

                self.tconv_layers.append(
                    TConvNd(in_channels=current_tconv_in_channels, out_channels=target_layer_out_channels,
                            kernel_size=layer_kernel, stride=layer_stride,
                            padding=layer_padding, output_padding=layer_output_padding)
                )
                if i < num_tconv_layers - 1:  # BN for intermediate layers
                    self.tbn_layers.append(BatchNormNd(target_layer_out_channels))

                current_tconv_in_channels = target_layer_out_channels

                # Simulate shape change for logging
                sim_D_prev, sim_H_prev, sim_W_prev = sim_D, sim_H, sim_W  # Store for logging
                if self.conv_dim == 3 and sim_D is not None:
                    sim_D = (sim_D - 1) * layer_stride[0] - 2 * layer_padding[0] + layer_kernel[0] + layer_output_padding[0]
                    sim_H = (sim_H - 1) * layer_stride[1] - 2 * layer_padding[1] + layer_kernel[1] + layer_output_padding[1]
                    sim_W = (sim_W - 1) * layer_stride[2] - 2 * layer_padding[2] + layer_kernel[2] + layer_output_padding[2]
                    print(
                        f"  Layer {i} TConv3D: C_out={target_layer_out_channels}, in_shape(D,H,W)=({sim_D_prev},{sim_H_prev},{sim_W_prev}), "
                        f"K={layer_kernel}, S={layer_stride}, P={layer_padding}, OP={layer_output_padding} "
                        f"-> sim_shape (D,H,W): ({sim_D}, {sim_H}, {sim_W})")
        else:
            current_channels = self.C_from_encoder
            # conv_dim == 2 (Simplified, adapt for specific 2D upsampling targets if needed)
            for i, target_out_channels in enumerate(self.tconv_output_channel_targets):
                        kernel_size = []
                        kernel_size.extend([self.decoder_hw_kernels_config[i]]*2)
                        padding = tuple(k // 2 for k in kernel_size)
                        stride = []
                        stride.extend([1, 1])
                        self.tconv_layers.append(
                            TConvNd(in_channels=current_channels, out_channels=target_out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding)
                        )
                        # Add BN for intermediate layers. Not for the last TConv if it produces final `output_channels`.
                        if i < num_tconv_layers - 1:  # Any layer that is NOT the last one
                            self.tbn_layers.append(BatchNormNd(target_out_channels))
                        # elif target_out_channels != output_channels:  # Last TConv, but it does NOT produce final `output_channels` (unusual)
                        #     self.tbn_layers.append(BatchNormNd(target_out_channels))
                        # Else (last TConv layer AND target_out_channels == output_channels): No BN.

                        current_channels = target_out_channels
        final_act_str = config.get('decoder_final_activation', 'sigmoid').lower()
        if final_act_str not in ACTIVATION_MAP:
            print(f"Warning: Unknown final_activation '{final_act_str}'. Using 'identity'.")
            final_act_str = 'identity'
        self.final_activation = ACTIVATION_MAP[final_act_str]()

    @staticmethod
    def _calculate_padding_and_op(dim_in, dim_target, stride, kernel_size, ideal_padding):
        """
        Calculates padding and output_padding to achieve target_dim.
        dim_target = (dim_in - 1) * stride - 2*P + K + OP
        Rearranging: 2P - OP = (dim_in - 1) * stride + K - dim_target
        """
        val_to_match = (dim_in - 1) * stride + kernel_size - dim_target

        best_P = None
        best_OP = None
        min_abs_diff_from_ideal_P = float('inf')

        for op_candidate in range(stride):  # OP is {0, ..., stride-1}
            # 2P = val_to_match + op_candidate
            if (val_to_match + op_candidate) % 2 == 0:
                P_candidate = (val_to_match + op_candidate) // 2
                if P_candidate >= 0:
                    # This is a valid solution (P >=0, op_candidate is valid)
                    abs_diff = abs(P_candidate - ideal_padding)

                    if abs_diff < min_abs_diff_from_ideal_P:
                        min_abs_diff_from_ideal_P = abs_diff
                        best_P = P_candidate
                        best_OP = op_candidate
                    elif abs_diff == min_abs_diff_from_ideal_P:
                        # Tie-breaking: prefer smaller OP if P_candidate results in same diff
                        if best_OP is None or op_candidate < best_OP:
                            # This check is mostly to ensure the first found (smallest OP for a given diff) is kept
                            best_P = P_candidate  # Redundant if min_abs_diff_from_ideal_P hasn't changed P
                            best_OP = op_candidate
        return best_P, best_OP

    def forward(self, z):
        x = z
        if self.use_decoder_fc:
            if self.fc is None:
                raise RuntimeError("Decoder: use_decoder_fc is True, but self.fc is None.")
            x = self.fc(x)

            if self.conv_dim == 2:
                target_shape = (x.size(0), self.C_from_encoder, self.H_to_reconstruct, self.W_to_reconstruct)
            else:  # conv_dim == 3
                target_shape = (x.size(0), self.C_from_encoder, self.D_to_reconstruct, self.H_to_reconstruct,
                                self.W_to_reconstruct)
            x = x.view(*target_shape)
        else:
            if self.conv_dim == 3:
                x = x.unsqueeze(1)
        for i in range(len(self.tconv_layers)):
            x = self.tconv_layers[i](x)
            if i < len(self.tbn_layers):  # Apply BN if it exists for this layer
                x = self.tbn_layers[i](x)

            # Apply activation for all intermediate layers, or if last layer has BN
            if i < len(self.tconv_layers) - 1:
                x = self.activation_fn(x)
            elif len(self.tbn_layers) == len(self.tconv_layers) and i == len(self.tconv_layers) - 1:
                # If last layer also had BN (less common for direct output but possible)
                x = self.activation_fn(x)
        if self.conv_dim == 3:
            x = x.squeeze(1)
        reconstructed_x = self.final_activation(x)
        return reconstructed_x

class SugarPredictorLatent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_layer = ACTIVATION_MAP[config.get('predictor_activation', 'relu')]()
        self.cls = config.get('cls', False)
        layers = []
        latent_dim = config['latent_dim']
        in_features = latent_dim
        for hidden_size in config['predictor_hidden_sizes']:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(self.activation_layer)
            layers.append(nn.Dropout(config.get('dropout', 0.3)))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, 1))  # Predicting one value (e.g., brix)
        self.net = nn.Sequential(*layers)
        if self.cls:
            cls_layers = []
            in_features = latent_dim
            for hidden_size in config['predictor_hidden_sizes']:
                cls_layers.append(nn.Linear(in_features, hidden_size))
                cls_layers.append(self.activation_layer)
                cls_layers.append(nn.Dropout(config.get('dropout', 0.3)))
                in_features = hidden_size
            cls_layers.append(nn.Linear(in_features, 2))  # Predicting one value (e.g., background)
            self.cls_net = nn.Sequential(*cls_layers)
            acidity_layers = []
            in_features = latent_dim
            for hidden_size in config['predictor_hidden_sizes']:
                acidity_layers.append(nn.Linear(in_features, hidden_size))
                acidity_layers.append(self.activation_layer)
                acidity_layers.append(nn.Dropout(config.get('dropout', 0.3)))
                in_features = hidden_size
            acidity_layers.append(nn.Linear(in_features, 1))  # Predicting one value (e.g., acidity)
            self.acidity_net = nn.Sequential(*acidity_layers)

    def forward(self, latent_code):
        if self.cls:
            return self.cls_net(latent_code), self.net(latent_code), self.acidity_net(latent_code)
        return self.net(latent_code)


class DomainDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_layer = ACTIVATION_MAP[config.get('activation', 'relu')]()
        layers = []
        in_features = config['latent_dim']
        for hidden_size in config['discriminator_hidden_sizes']:
            layers.append(nn.Linear(in_features, hidden_size))
            # Consider LeakyReLU for discriminators
            layers.append(
                nn.LeakyReLU(0.2, inplace=True) if config.get('activation') == 'leaky_relu' else self.activation_layer)
            layers.append(nn.Dropout(config.get('dropout', 0.3)))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, config['num_domains']))
        self.net = nn.Sequential(*layers)

    def forward(self, latent_code):
        latent_code = latent_code.view(latent_code.size(0), -1)
        return self.net(latent_code)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std



class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_coeff):
        ctx.lambda_coeff = lambda_coeff
        return x.view_as(x)  # or x.clone() if view_as causes issues

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is the gradient from the subsequent layer (discriminator)
        # We reverse it and scale by lambda for the preceding layer (encoder)
        # For the GRL's own parameters (if any), grad would be None
        return grad_output.neg() * ctx.lambda_coeff, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_coeff=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_coeff = lambda_coeff

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_coeff)

    def update_lambda(self, new_lambda):  # Optional: if you want to anneal lambda
        self.lambda_coeff = new_lambda



def transfer_loop(common_loader_params, encoder, predictor, ae_config, config, wandb, epoch, dataset):
    # --- Evaluation loop ---
    common_loader_params['labo'] = False #if not dataset == 0 else True
    common_loader_params['sweep'] = False if dataset == 0 else True # False if dataset == 1 else (True if dataset == 2 else False)
    common_loader_params['extendedData'] = False
    common_loader_params['extendedDataReferences'] = False
    del common_loader_params['probBackground']
    encoder.eval()
    predictor.eval()
    dataset = 1 if dataset == 0 else 2

    for backgroundB in range(2 if config['dataLoader.background'] else 1):
        predicted_reg_all_cls = []
        transfer_predicted_reg_all_cls = []
        transfer_target_reg_all_brix = []
        transfer_target_reg_all_acid = []
        transfer_reg_all_brix = []
        transfer_reg_all_acid = []
        transfer_reg_all_cls = []
        transfer_index = sorted(list(set(range(360)) - {318, 319, 320, 321, 322, 323}))
        transferDataLoader = grape_dataLoader(**common_loader_params,
                                              Indexs=transfer_index,
                                              probBackground=0.5 if backgroundB else 0)
        transferData = Subset(transferDataLoader, transfer_index)
        transferloader = DataLoader(transferData, batch_size=4, shuffle=False, num_workers=1)

        transferIter = iter(transferloader)
        with torch.no_grad():
            for batch_idx in range(len(transferIter)):
                data = next(transferIter)
                images = data['image'].to(device).float()
                sugar_labels_true = data['brix'].to(device).float()
                cls_labels_true = data['cls'].to(device).long().squeeze(1)
                acid_labels_true = data['acid'].to(device).float()

                transfer_reg_all_cls.append(cls_labels_true)
                if ae_config['variational'] and ae_config['use_encoder_fc']:
                    latent_code_eval, _, conv_features = encoder(images)  # Use mu for eval
                else:
                    latent_code_eval, conv_features = encoder(images)

                # Sugar prediction
                if config['ae.predictor_input_type']:
                    sugar_preds = predictor(conv_features)
                else:
                    sugar_preds = predictor(latent_code_eval)
                # sugar_preds = models[modelIt][1](latent_code_eval)
                if config['dataLoader.background']:
                    predicted_reg_all_cls.append(sugar_preds[0].squeeze(1))
                    transfer_predicted_reg_all_cls.append(sugar_preds[0].squeeze(1))
                    regression_mask = (cls_labels_true == 1).float()
                    if regression_mask.sum() > 0:
                        output_reg1 = sugar_preds[1][regression_mask == 1]
                        target_reg1 = sugar_labels_true[regression_mask == 1]

                        output_reg2 = sugar_preds[2][regression_mask == 1]
                        target_reg2 = acid_labels_true[regression_mask == 1]

                        transfer_target_reg_all_brix.append(target_reg1)
                        transfer_target_reg_all_acid.append(target_reg2)
                        transfer_reg_all_brix.append(output_reg1)
                        transfer_reg_all_acid.append(output_reg2)
                else:
                    output_reg1 = sugar_preds.squeeze(1)
                    transfer_reg_all_brix.append(output_reg1)
                    transfer_target_reg_all_brix.append(sugar_labels_true)

                if not backgroundB:
                    predicted_value_brix = torch.cat(transfer_reg_all_brix[-1:], dim=0).cpu().detach()
                    target_reg_brix = torch.cat(transfer_target_reg_all_brix[-1:], dim=0).cpu().detach()
                    target_value_brix = target_reg_brix.squeeze()
                    r_two, mse = get_metrics(predicted_value_brix.squeeze(), target_value_brix)
                    wandb.log({
                               f'transfer_slice': batch_idx,
                               f'{dataset}_transfer_Intermediate_r_two_brix{"_" + str(epoch)}': r_two,
                               f'{dataset}_transfer_Intermediate_mse_brix{"_" + str(epoch)}': mse,
                               f'{dataset}_transfer_Intermediate_devation_brix{"_" + str(epoch)}': np.sqrt(
                                   mse),
                               })
                if not backgroundB and config['dataLoader.background']:
                    predicted_value_acid = torch.cat(transfer_reg_all_acid[-1:], dim=0).cpu().detach()
                    target_reg = torch.cat(transfer_target_reg_all_acid[-1:], dim=0).cpu().detach()
                    target_value_acid = target_reg.squeeze()
                    r_two_acid, mse_acid = get_metrics(predicted_value_acid.squeeze(),
                                                       target_value_acid)
                    wandb.log({
                               f'transfer_slice': batch_idx,
                               f'{dataset}_transfer_Intermediate_r_two_acid{"_" + str(epoch)}': r_two_acid,
                               f'{dataset}_transfer_Intermediate_mse_acid{"_" + str(epoch)}': mse_acid,
                               f'{dataset}_transfer_Intermediate_devation_acid{"_" + str(epoch)}': np.sqrt(
                                   mse_acid),
                               })
                if backgroundB:
                    predicted_cls = torch.cat(transfer_predicted_reg_all_cls[-1:], dim=0).cpu().detach()
                    target_cls = torch.cat(transfer_reg_all_cls[-1:], dim=0).cpu().detach()
                    correct_predictions = (predicted_cls.argmax(dim=1) == target_cls).sum().item()
                    total_predictions = target_cls.size(0)
                    classification_accuracy = correct_predictions / total_predictions
                    wandb.log({
                               f'transfer_slice': batch_idx,
                               f'{dataset}_transfer_Intermediate_classification_accuracy{"_" + str(epoch)}': classification_accuracy,
                               })

            if not backgroundB:
                predicted_value_brix = torch.cat(transfer_reg_all_brix, dim=0).cpu().detach()
                target_reg_brix = torch.cat(transfer_target_reg_all_brix, dim=0).cpu().detach()
                target_value_brix = target_reg_brix.squeeze()
                r_two, mse = get_metrics(predicted_value_brix.squeeze(), target_value_brix)
                wandb.log({
                           'dataset': dataset,
                           f'{dataset}_transfer_r_two_brix{"_" + str(epoch)}': r_two,
                           f'{dataset}_transfer_mse_brix{"_" + str(epoch)}': mse,
                           f'{dataset}_transfer_devation_brix{"_" + str(epoch)}': np.sqrt(
                               mse),
                           f'{dataset}_transfer_r_two_brix': r_two,
                           })
                transfer_brix = r_two
            if not backgroundB and config['dataLoader.background']:
                predicted_value_acid = torch.cat(transfer_reg_all_acid, dim=0).cpu().detach()
                target_reg = torch.cat(transfer_target_reg_all_acid, dim=0).cpu().detach()
                target_value_acid = target_reg.squeeze()
                r_two_acid, mse_acid = get_metrics(predicted_value_acid.squeeze(), target_value_acid)
                wandb.log({
                           'dataset': dataset,
                           f'{dataset}_transfer_r_two_acid{"_" + str(epoch)}': r_two_acid,
                           f'{dataset}_transfer_mse_acid{"_" + str(epoch)}': mse_acid,
                           f'{dataset}_transfer_devation_acid{"_" + str(epoch)}': np.sqrt(
                               mse_acid),
                           f'{dataset}_transfer_r_two_acid': r_two_acid,

                        })
            if backgroundB:
                predicted_cls = torch.cat(transfer_predicted_reg_all_cls, dim=0).cpu().detach()
                target_cls = torch.cat(transfer_reg_all_cls, dim=0).cpu().detach()
                correct_predictions = (predicted_cls.argmax(dim=1) == target_cls).sum().item()
                total_predictions = target_cls.size(0)
                classification_accuracy = correct_predictions / total_predictions
                wandb.log({
                           'dataset': dataset,
                           f'{dataset}_transfer_classification_accuracy{"_" + str(epoch)}': classification_accuracy,
                           f'{dataset}_transfer_classification_accuracy': classification_accuracy,
                           })
        best_r2_in_sweep = get_best_metric_from_sweep(wandb,
                                                      f'{dataset}_transfer_r_two_brix')

        if transfer_brix > best_r2_in_sweep:
            print(
                f"🎉 New best model in sweep! R²: {best_r2_in_sweep:.4f} (previous best: {best_r2_in_sweep:.4f})")

            plot_predictions2(predicted_value_brix, target_reg_brix, wandb, 'Brix')
            if config.get('dataLoader.background', True):
                plot_predictions2(predicted_value_acid, target_value_acid, wandb, 'Acid')
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'sugar_predictor_state_dict': predictor.state_dict(),
            },
                f'/project_ghent/grapePaper/models/ae_predictor_epoch_{epoch}_R2_{transfer_brix:.3f}_dataset_{dataset}_runID_{wandb.run.id}.pth')
    return transfer_brix

def transfer_loop_forBaseline(common_loader_params,predictor, wandb, epoch, dataset, label):
    # --- Evaluation loop ---
    common_loader_params['labo'] = False #if not dataset == 0 else True
    common_loader_params['sweep'] = False if dataset == 0 else True # (True if dataset == 2 else False)
    common_loader_params['extendedData'] = False
    common_loader_params['extendedDataReferences'] = False
    logsep = common_loader_params.get('logsep', False)
    del common_loader_params['probBackground']
    predictor.eval()
    backgroundB = 0 if label != 2 else 1
    transfer_target_reg_all_brix = []
    transfer_reg_all_brix = []
    transfer_predicted_reg_all_cls = []
    transfer_reg_all_cls = []
    transfer_index = sorted(list(set(range(360)) - {318, 319, 320, 321, 322, 323}))
    transferDataLoader = grape_dataLoader(**common_loader_params,
                                          Indexs=transfer_index,
                                          probBackground=0.5 if backgroundB else 0)
    transferData = Subset(transferDataLoader, transfer_index)
    transferloader = DataLoader(transferData, batch_size=18, shuffle=False, num_workers=1)
    dataset = 1 if dataset == 0 else 2
    transferIter = iter(transferloader)
    with torch.no_grad():
        for batch_idx in range(len(transferIter)):
            data = next(transferIter)
            images_raw = data['image'].to(device).float()
            sugar_labels_true = data['brix' if label else 'acid'].to(device).float()
            cls_labels_true = data['cls'].to(device).long().squeeze(1)
            transfer_reg_all_cls.append(cls_labels_true)
            if not logsep:
                # --- ADJUSTMENT 5: Reshape data for evaluation as well ---
                B = images_raw.size(0)
                images = images_raw.view(B, 224, 64).transpose(1, 2)  # (B, P, S)

                d_real_output, task_preds_real, domain_preds_real = predictor(images)
            else:
                task_preds_real = predictor(images_raw)
            if label != 2:
                transfer_reg_all_brix.append(task_preds_real.cpu())
                transfer_target_reg_all_brix.append(sugar_labels_true.cpu())

                predicted_value_brix = torch.cat(transfer_reg_all_brix[-1:], dim=0).cpu().detach()
                target_reg_brix = torch.cat(transfer_target_reg_all_brix[-1:], dim=0).cpu().detach()
                target_value_brix = target_reg_brix.squeeze()
                r_two, mse = get_metrics(predicted_value_brix.squeeze(), target_value_brix)
                wandb.log({
                    f'transfer_slice': batch_idx,
                    f'{dataset}_transfer_Intermediate_r_two_{"brix" if label == 0 else "acid"}{"_" + str(epoch)}': r_two,
                    f'{dataset}_transfer_Intermediate_mse_{"brix" if label == 0 else "acid"}{"_" + str(epoch)}': mse,
                    f'{dataset}_transfer_Intermediate_devation_{"brix" if label == 0 else "acid"}{"_" + str(epoch)}': np.sqrt(
                        mse),
                })
            else:
                transfer_predicted_reg_all_cls.append(task_preds_real.squeeze(1))
                predicted_cls = torch.cat(transfer_predicted_reg_all_cls[-1:], dim=0).cpu().detach()
                target_cls = torch.cat(transfer_reg_all_cls[-1:], dim=0).cpu().detach()
                correct_predictions = (predicted_cls.argmax(dim=1) == target_cls).sum().item()
                total_predictions = target_cls.size(0)
                classification_accuracy = correct_predictions / total_predictions
                wandb.log({
                    f'transfer_slice': batch_idx,
                    f'{dataset}_transfer_Intermediate_classification_accuracy{"_" + str(epoch)}': classification_accuracy,
                })
        if label !=2 :
            predicted_value_brix = torch.cat(transfer_reg_all_brix, dim=0).cpu().detach()
            target_reg_brix = torch.cat(transfer_target_reg_all_brix, dim=0).cpu().detach()
            target_value_brix = target_reg_brix.squeeze()
            r_two, mse = get_metrics(predicted_value_brix.squeeze(), target_value_brix)
            wandb.log({
                'dataset': dataset,
                f'{dataset}_transfer_r_two_{"brix" if label == 0 else "acid"}{"_" + str(epoch)}': r_two,
                f'{dataset}_transfer_mse_{"brix" if label == 0 else "acid"}{"_" + str(epoch)}': mse,
                f'{dataset}_transfer_devation_{"brix" if label == 0 else "acid"}{"_" + str(epoch)}': np.sqrt(
                    mse),
            })
            transfer_brix = r_two
        else:
            predicted_cls = torch.cat(transfer_predicted_reg_all_cls, dim=0).cpu().detach()
            target_cls = torch.cat(transfer_reg_all_cls, dim=0).cpu().detach()
            correct_predictions = (predicted_cls.argmax(dim=1) == target_cls).sum().item()
            total_predictions = target_cls.size(0)
            classification_accuracy = correct_predictions / total_predictions
            wandb.log({
                'dataset': dataset,
                f'{dataset}_transfer_classification_accuracy{"_" + str(epoch)}': classification_accuracy,
            })
            transfer_brix = classification_accuracy
    return transfer_brix

# 1. Mapping Network (F) - Creates domain-specific style codes
class ConditionalPath(nn.Module):
    """A helper module representing one of the Y conditional paths."""

    def __init__(self, p_dim, s_dim):
        super().__init__()
        self.p_dim = p_dim
        self.s_dim = s_dim

        # This layer takes the flattened shared features (2S * 2P) and maps them
        # to the shape needed before the final layer (P * 2S).
        self.layer1 = nn.Linear(2 * p_dim, p_dim)

        # This layer operates on the last dimension (2S) to produce the final S.
        self.layer2 = nn.Linear(2 * s_dim, s_dim)

    def forward(self, x):
        # x is the shared feature tensor of shape (B, 2S, 2P)
        B = x.size(0)

        # Flatten for the first linear layer
        x = x.view(B, 2*self.s_dim, 2*self.p_dim)
        h1 = F.relu(self.layer1(x))
        h1_t = h1.transpose(-2, -1)

        # Reshape to (B, P, 2S) to prepare for the final layer
        # h1 = h1_t.view(B, self.p_dim, 2*self.s_dim)

        # Final layer to get the style tensor shape (B, P, S)
        style_tensor = F.relu(self.layer2(h1_t))

        return style_tensor


class MappingNetworkF(nn.Module):
    """
    Implementation of the Mapping Network with separate paths for each domain.
    """

    def __init__(self, p_dim, s_dim, y_dim):
        super().__init__()
        self.p_dim = p_dim
        self.s_dim = s_dim
        self.y_dim = y_dim

        # --- 1. Shared Encoder Path (Same as before) ---
        self.shared_layer1 = nn.Linear(s_dim, 2 * s_dim)
        self.shared_layer2 = nn.Linear(p_dim, 2 * p_dim)

        # --- 2. Y Separate Conditional Decoder Paths ---
        # We create a list of 'Y' independent 'ConditionalPath' modules.
        # Each module will have its own unique weights.
        self.conditional_paths = nn.ModuleList(
            [ConditionalPath(p_dim, s_dim) for _ in range(y_dim)]
        )

    def forward(self, z, y=torch.zeros(2,2).to(device)):
        B, P, S = z.shape
        _B, Y = y.shape
        assert B == _B and P == self.p_dim and S == self.s_dim and Y == self.y_dim

        # --- 1. Shared Encoder ---
        h1 = F.relu(self.shared_layer1(z))
        h1_t = h1.transpose(-2, -1)
        h_shared = F.relu(self.shared_layer2(h1_t))  # Shape: (B, 2S, 2P)

        # --- 2. Generate all Y possible style tensors ---
        # Pass the shared features through each domain-specific path
        all_domain_styles = [path(h_shared) for path in self.conditional_paths]

        # Stack the results along a new 'domain' dimension
        # List of Y tensors of shape (B, P, S) -> One tensor of shape (B, Y, P, S)
        h_cond_final = torch.stack(all_domain_styles, dim=1)

        # --- 3. Final Selection ---
        # Broadcast the one-hot y vector to select the correct style
        y_broadcast = y.view(B, Y, 1, 1)

        # Multiply and sum to perform the selection
        s = torch.sum(h_cond_final * y_broadcast, dim=1)  # Shape: (B, P, S)

        return s


# 2. Generator (G) - Your modified Decoder
# This class re-uses your DecoderAE_3D logic but adapts the forward pass.
# I've kept your complex DecoderAE_3D code as is, just renaming it and changing the forward method.
class GeneratorG(nn.Module):
    """
    A PyTorch implementation of the Generator architecture shown in the diagram.
    """

    def __init__(self):
        super().__init__()

        # We use padding to ensure the spatial dimensions (P, S) are preserved
        # after each convolution. For a kernel of size K, padding = (K-1)/2.
        # K=5 -> padding=2
        # K=3 -> padding=1
        # K=1 -> padding=0

        # --- Encoder Path (leading to the bottleneck) ---
        self.encoder = nn.Sequential(
            # Input: (B, 2, P, S) -> Output: (B, 4, P, S)
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, padding=2),
            nn.InstanceNorm2d(4),
            nn.ReLU(inplace=True),
            # Input: (B, 4, P, S) -> Output: (B, 16, P, S) (This is the bottleneck)
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            # Input: (B, 4, P, S) -> Output: (B, 16, P, S) (This is the bottleneck)
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # --- Main Decoder Path (left branch after bottleneck) ---
        self.decoder_main = nn.Sequential(
            # Input: (B, 16, P, S) -> Output: (B, 4, P, S)
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(4),
            nn.ReLU(inplace=True),
            # Input: (B, 4, P, S) -> Output: (B, 1, P, S)
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=5, padding=2),
            nn.InstanceNorm2d(1),
            nn.ReLU(inplace=True)
        )


    def forward(self, x, s = torch.zeros(2,64,224).to(device)):
        """
        Forward pass for the Generator.
        Args:
            x (Tensor): The primary input tensor, shape [B, P, S] or [B, 1, P, S].
            s (Tensor): The secondary input tensor, shape [B, P, S] or [B, 1, P, S].
        Returns:
            Tensor: The generated output, shape [B, P, S].
        """
        # Ensure inputs have a channel dimension for 2D convolutions
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Shape: [B, 1, P, S]
        if s.dim() == 3:
            s = s.unsqueeze(1)  # Shape: [B, 1, P, S]

        # 1. Initial concatenation (bottom circle in the diagram)
        net_input = torch.cat([x, s], dim=1)  # Shape: [B, 2, P, S]
        # 2. Pass through the encoder to get the bottleneck features
        bottleneck = self.encoder(net_input)  # Shape: [B, 16, P, S]

        # 3. Pass the bottleneck through the two parallel decoder paths
        main_path_out = self.decoder_main(bottleneck)  # Shape: [B, 1, P, S]

        # 4. Combine the paths via element-wise addition (top circle in the diagram)
        output = main_path_out  # Shape: [B, 1, P, S]

        # Note: The diagram does not show a final activation function like Tanh or Sigmoid.
        # If your data is normalized to [-1, 1], you might want to add a Tanh activation here.
        # Example: output = torch.tanh(output)

        # 5. Squeeze the channel dimension to match the diagram's final output shape (B, P, S)
        return output.squeeze(1)


# 3. Discriminator (D) - Multi-headed network
# class DiscriminatorD(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         # Shared feature extractor
#         in_channels = 224 if config.get('conv_dim', 2) == 2 else 1
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         # This needs to be calculated based on your input image size and the strides above
#         # Example for a 224x8x8 input -> after convs -> 256 x 1 x 1
#         # For simplicity, let's assume we flatten to this size
#         feature_dim = 256
#
#         # 1. Real vs. Fake head
#         self.real_fake_head = nn.Linear(feature_dim, 1)
#
#         # 2. Domain classification head
#         self.domain_head = nn.Linear(feature_dim, config['num_domains'])
#
#         # 3. Task prediction head (your 'sugar predictor')
#         # This head predicts brix, acid, and background class
#         self.task_head_cls = nn.Linear(feature_dim, 2)  # For background/foreground classification
#         self.task_head_brix = nn.Linear(feature_dim, 1)  # For brix regression
#         self.task_head_acid = nn.Linear(feature_dim, 1)  # For acid regression
#         self.background_regression = config['dataLoader.background']
#
#     def forward(self, x):
#         # Get shared features
#         features = self.feature_extractor(x)
#         features = features.view(x.size(0), -1)  # Flatten
#
#         # Get outputs from each head
#         out_real_fake = self.real_fake_head(features)
#         out_domain = self.domain_head(features)
#
#         if self.background_regression:
#             out_cls = self.task_head_cls(features)
#             out_brix = self.task_head_brix(features)
#             out_acid = self.task_head_acid(features)
#             return out_real_fake, out_domain, (out_cls, out_brix, out_acid)
#         else:
#             out_brix = self.task_head_brix(features)
#             # You might want a separate head for acid if not using the background logic
#             return out_real_fake, out_domain, out_brix

# 1. Gradient Reversal Layer (for the adversarial path)
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# 2. Re-zero module (for the residual connection)
class Rezero(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.alpha

# 3. A basic TCN block with Gated Linear Unit (GLU)
# Note: The diagram's "TCN Conv1d(K=3,5,7)" is ambiguous. This implementation
# uses a single dilated convolution, which is a common TCN building block.
# For simplicity, we use one kernel size.
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1):
        super().__init__()
        # The Conv1d needs to output 2*out_channels for the GLU
        padding = (kernel_size - 1) * dilation // 2
        self.conv = weight_norm(nn.Conv1d(in_channels, 2 * out_channels, kernel_size,
                                          padding=padding, dilation=dilation))

    def forward(self, x):
        x = self.conv(x)
        # GLU (GuLe in the diagram) activation
        # It splits the channels in half, applies tanh to one half, sigmoid to the other,
        # and multiplies them. F.glu does this.
        x = F.glu(x, dim=1) # dim=1 is the channel dimension
        return x

class ReverseLayerF(Function):
    """
    A PyTorch autograd Function for the Gradient Reversal Layer.
    This is the new-style implementation that uses static methods.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        # Store the alpha value for the backward pass
        ctx.alpha = alpha
        # The forward pass is an identity function
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient and scale it by alpha
        # The gradient for x is grad_output.neg() * alpha
        # The gradient for alpha is None, as we don't want to backprop through it
        output = grad_output.neg() * ctx.alpha
        return output, None



class DiscriminatorD(nn.Module):
    def __init__(self, p_dim=64, s_dim=224, k_dim=1, y_dim=2, dropout_rate=0.1):
        """
        Implementation of the Discriminator from the diagram.
        Args:
            p_dim (int): The 'P' dimension from the input shape (B, P, S).
            s_dim (int): The 'S' dimension from the input shape (B, P, S).
            k_dim (int): The output dimension for the 'k' head.
            y_dim (int): The output dimension for the 'y' head (adversarial).
            dropout_rate (float): Dropout rate for the main head.
        """
        super().__init__()
        self.p_dim = p_dim
        self.s_dim = s_dim

        # --- 1. TCN Feature Extractor ---
        # The TCN block keeps the P dimension the same size
        self.tcn_block1 = TCNBlock(in_channels=p_dim, out_channels=p_dim, kernel_size=3)
        self.tcn_block2 = TCNBlock(in_channels=p_dim, out_channels=p_dim, kernel_size=5)
        self.tcn_block3 = TCNBlock(in_channels=p_dim, out_channels=p_dim, kernel_size=7)

        # --- 2. Self-Attention Block ---
        self.pre_attention_norm = nn.LayerNorm([p_dim, s_dim])
        self.q_proj = nn.Linear(s_dim, s_dim)
        self.k_proj = nn.Linear(s_dim, s_dim)
        self.v_proj = nn.Linear(s_dim, s_dim)
        self.attention_softmax = nn.Softmax(dim=-1)

        # --- 3. Re-zero Residual Connection ---
        self.rezero = Rezero()
        self.rezeroInput = Rezero()
        # --- 4. Main Path to Real/Fake and 'k' outputs ---
        self.main_path = nn.Sequential(
            nn.LayerNorm([p_dim, s_dim]),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(0.2, inplace=True)  # "RuLe" -> Rectified Unit Linear..
        )
        # This Conv1d is inferred to achieve the (B, 4, S) shape from (B, P, S)
        # We treat P as channels and S as sequence length
        self.main_path_conv = nn.Conv1d(in_channels=p_dim, out_channels=4, kernel_size=1)

        # Final linear heads for the main path
        self.head_real_fake = nn.Linear(4 * s_dim, 1)
        self.head_k = nn.Linear(4 * s_dim, k_dim)
        self.pooling_block = nn.Sequential(
            # First, change channels from 1 to 4 using Conv2d
            # nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1),
            # Then, pool across the P dimension to get (B, 4, 1, S)
            nn.AvgPool2d(kernel_size=(16, 1))
        )
        # --- 5. Adversarial Path to 'y' output ---
        # This block contains the GRL and the projection from (B, P, S) to (B, 4S)
        # self.RLF = ReverseLayerF()
        self.adversarial_path = nn.Sequential(
            nn.Linear(4 * s_dim, s_dim),
            nn.LayerNorm(s_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.head_y = nn.Linear(s_dim, y_dim)

    def forward(self, x, grl_alpha=1.0):
        """
        Args:
            x (Tensor): Input tensor of shape (B, P, S).
            grl_alpha (float): The alpha parameter for the Gradient Reversal Layer.
        """
        b, p, s = x.shape
        assert p == self.p_dim and s == self.s_dim, "Input dimensions do not match."

        # 1. TCN Block
        tcn_out1 = self.tcn_block1(x)  # Shape: (B, P, S)
        tcn_out2 = self.tcn_block2(tcn_out1)  # Shape: (B, P, S)
        tcn_out = self.tcn_block3(tcn_out2)  # Shape: (B, P, S)

        # 2. Self-Attention Block
        h_norm = self.pre_attention_norm(tcn_out)
        q = self.q_proj(h_norm)  # Shape: (B, P, S)
        k = self.k_proj(h_norm)  # Shape: (B, P, S)
        v = self.v_proj(h_norm)  # Shape: (B, P, S)

        # Attention scores: (B, P, S) @ (B, S, P) -> (B, P, P)
        attention_scores = self.attention_softmax(torch.matmul(q, k.transpose(-2, -1)))

        # Attention output: (B, P, P) @ (B, P, S) -> (B, P, S)
        attention_out = torch.matmul(attention_scores, v)

        # 3. Residual Connection with Re-zero
        h = attention_out + self.rezero(h_norm) + self.rezeroInput(x) # Shape: (B, P, S)

        # --- 4. Main Path ---
        h_main = self.main_path(h)  # Shape: (B, P, S)

        # The diagram shows AvgPool2d and then (B, 4, S). This is likely a simplification.
        # A 1D conv on the P-dimension is a more direct way to achieve this.
        h_main = self.main_path_conv(h_main)  # Shape: (B, 4, S)
        # h_pool_input = h_main.unsqueeze(1)
        # Apply pooling block: (B, 1, P, S) -> (B, 4, 1, S)
        # h_pooled = self.pooling_block(h_main)
        # Flatten for linear layers: (B, 4, 1, S) -> (B, 4*S)
        h_final = h_main.view(b, -1)

        out_real_fake = self.head_real_fake(h_final)  # Shape: (B, 1)
        out_k = self.head_k(h_final)  # Shape: (B, K)

        # --- 5. Adversarial Path ---
        # The GRL path starts from the TCN output
        rlf_out = ReverseLayerF.apply(h_final, grl_alpha)
        # rlf_out = self.RLF(h_final, grl_alpha)
        h_adv = self.adversarial_path(rlf_out)  # Shape: (B, S)
        out_y = self.head_y(h_adv)  # Shape: (B, Y)

        return out_real_fake, out_k, out_y




class LOGSEP:
    def __init__(self, m=3, n=12, lambda_reg=1e-6):
        self.m = m  # Number of illumination basis vectors
        self.n = n  # Number of reflectance basis vectors
        self.lambda_reg = lambda_reg
        self.E_basis = None
        self.S_basis = None
        self.T_regression = None

    def _prepare_spectra(self, spectra):
        """Take log and handle small values."""
        spectra = np.maximum(spectra, 1e-8)
        return np.log(spectra)

    def train(self, training_illuminations, training_reflectances):
        """
        Phase 1: Learn basis and regression matrices.
        - training_illuminations: (num_bands, num_samples)
        - training_reflectances: (num_bands, num_samples)
        """
        # Step 1: Prepare data
        log_E_train = self._prepare_spectra(training_illuminations)
        log_S_train = self._prepare_spectra(training_reflectances)

        # Step 2: Learn Basis Vectors via SVD
        U_e, _, _ = svd(log_E_train, full_matrices=False)
        self.E_basis = U_e[:, :self.m]

        U_s, _, _ = svd(log_S_train, full_matrices=False)
        self.S_basis = U_s[:, :self.n]

        # Step 3: Build Regression Matrix T
        # A) Find ground truth coefficients
        eps_coeffs = self.E_basis.T @ log_E_train
        sig_coeffs = self.S_basis.T @ log_S_train

        # B) Create large coefficient matrices
        alpha_tilde = np.vstack([eps_coeffs, sig_coeffs])

        # C) Calculate T
        identity_reg = np.eye(alpha_tilde.shape[0])
        term_to_invert = alpha_tilde @ alpha_tilde.T - self.lambda_reg * identity_reg

        Q = eps_coeffs @ alpha_tilde.T @ inv(term_to_invert)
        R = sig_coeffs @ alpha_tilde.T @ inv(term_to_invert)
        self.T_regression = np.vstack([Q, R])

        print("Training complete. Basis and regression matrices are stored.")

    def separate(self, new_radiance):
        """
        Phase 2: Separate a new radiance spectrum.
        - new_radiance: (num_bands,)
        """
        if self.T_regression is None:
            raise RuntimeError("Model must be trained first.")

        # Step 4: Initial Separation
        C_new = self._prepare_spectra(new_radiance)

        M = self.E_basis.T @ self.E_basis
        P = self.S_basis.T @ self.S_basis
        N = self.E_basis.T @ self.S_basis

        A = np.block([[M, N], [N.T, P]])

        f = self.E_basis.T @ C_new
        g = self.S_basis.T @ C_new
        h = np.concatenate([f, g])

        x_initial = inv(A) @ h

        # Step 5: Refine with Regression
        x_refined = self.T_regression @ x_initial

        # Step 6: Reconstruct Spectra
        eps_refined = x_refined[:self.m]
        sig_refined = x_refined[self.m:]

        log_E_recovered = self.E_basis @ eps_refined
        log_S_recovered = self.S_basis @ sig_refined

        illumination_recovered = np.exp(log_E_recovered)
        reflectance_recovered = np.exp(log_S_recovered)

        return reflectance_recovered, illumination_recovered



def AE_modelTrainer(project, run_id, get_configs = False):
    modelMAPProject = {0: 'sweep_grapePaper_patchFieldSweep_V1.0', 1: 'sweep_grapePaper_patchFieldSweepCLS_V1.0'}
    modelMAP = {'mlp': ['mtisuavk', 'nesfx8xt'], 'cnn1d': ['sxssyfpc', 'cmuwju3r'], 'cnn2d': ['2fqklxj0', 'zqpn9bfk'],
                'transformer': ['g5izl0r1', 'o7fd2fs2'], 'cnn_transformer': ['n5xuu9br', 'i0xhss91']}
    modelMAPCLS = {'mlp': ['gwy2az30', 'gyel76vq'], 'cnn1d': ['eou0uvrv', '8z9cz01x'], 'cnn2d': ['y77e3g3x', '3s1rnq9f'],
                   'transformer': ['m4jrvz8t', '0vaqd3sa'], 'cnn_transformer': ['9mku6hbx', 'c9p9rs8j']}
    model_epoch_save = [25, 50, 75, 100, 150, 200, 250, 300, 350,400]
    MAX_EPOCHS = 201
    api = wandb.Api()
    run = api.run(f"/{project}/{run_id}")
    config = run.config
    wandb.init(
        project="grape_validatingModels",
        id=f"{project.split('_')[-1]}_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config
            )
    run = api.run(
        f"/{modelMAPProject[config['dataLoader.background']]}/{modelMAP[config['model.type']][config['model.iter']] if not config['dataLoader.background'] else modelMAPCLS[config['model.type']][config['model.iter']]}")
    optimalConfig = run.config
    run = api.run(
        f"/{modelMAPProject[config['dataLoader.background']]}/{modelMAP[config['ae.discriminator.model.type']][config['model.iter']] if not config['dataLoader.background'] else modelMAPCLS[config['ae.discriminator.model.type']][config['model.iter']]}")
    optimalConfig_disc = run.config
    run = api.run(
        f"/{modelMAPProject[config['dataLoader.background']]}/{modelMAP[config['ae.discriminator.model.type2']][config['model.iter']] if not config['dataLoader.background'] else modelMAPCLS[config['ae.discriminator.model.type2']][config['model.iter']]}")
    optimalConfig_disc2 = run.config

    # --- Standard Model Config (for non-AE models or if AE encoder uses parts of it) ---
    model_config = {
        # Use config['key.with.dots']
        'model_type': optimalConfig['model.type'],
        'num_bands': config['ae.encoder.channels'][-1],
        'patch_height': 8,
        'patch_width': 8,

        'AE': not config['ae.predictor_input_type'] and config.get('ae.use_encoder_fc', True),
        'conv_dim': config.get('ae.conv_dim', 2),
        # CNN params
        'shared_cnn_channels': optimalConfig['model.cnn.shared_channels'],
        'shared_cnn_kernels': optimalConfig['model.cnn.shared_kernels'],
        'shared_cnn_pools': optimalConfig['model.cnn.shared_pools'],
        'use_batchnorm': optimalConfig['model.cnn.use_batchnorm'],
        'dropout_shared': optimalConfig['model.cnn.dropout_shared'],  # Used by CNN and MLP
        'instanceNorm': config['model.instanceNorm'],
        # MLP params
        'shared_mlp_sizes': optimalConfig['model.mlp.shared_sizes'],

        # Transformer params
        'transformer_patch_size': optimalConfig['model.transformer.patch_size'],
        'transformer_embed_dim': optimalConfig['model.transformer.embed_dim'],
        'transformer_depth': optimalConfig['model.transformer.depth'],
        'transformer_heads': optimalConfig['model.transformer.heads'],
        'transformer_mlp_dim': optimalConfig['model.transformer.mlp_dim'],
        'transformer_dropout': optimalConfig['model.transformer.dropout'],
        'use_cls_token': optimalConfig['model.transformer.use_cls_token'],
        'use_pos_embed': optimalConfig['model.transformer.use_positional_encoding'],
        'transformer_stride': optimalConfig['model.transformer.stride'],
        'cls': config['dataLoader.background'],

        # Regression head params
        'regression_hidden_sizes': optimalConfig['model.regression.hidden_sizes'],
        'cls_hidden_sizes': optimalConfig.get('model.cls.hidden_sizes', False),
        'dropout_regression': optimalConfig['model.regression.dropout'],
        'use_multi_head': optimalConfig['model.regression.use_multihead'],
    }
    model_config_disc = {
        # Use config['key.with.dots']
        'model_type': optimalConfig_disc['model.type'],
        'num_bands': config['ae.encoder.channels'][-1],
        'patch_height': 8,
        'patch_width': 8,

        'AE': not config['ae.predictor_input_type'] and config.get('ae.use_encoder_fc', True),
        'conv_dim': config.get('ae.conv_dim', 2),
        # CNN params
        'shared_cnn_channels': optimalConfig_disc['model.cnn.shared_channels'],
        'shared_cnn_kernels': optimalConfig_disc['model.cnn.shared_kernels'],
        'shared_cnn_pools': optimalConfig_disc['model.cnn.shared_pools'],
        'use_batchnorm': optimalConfig_disc['model.cnn.use_batchnorm'],
        'dropout_shared': optimalConfig_disc['model.cnn.dropout_shared'],  # Used by CNN and MLP
        'instanceNorm': config['model.instanceNorm'],

        # MLP params
        'shared_mlp_sizes': optimalConfig_disc['model.mlp.shared_sizes'],

        # Transformer params
        'transformer_patch_size': optimalConfig_disc['model.transformer.patch_size'],
        'transformer_embed_dim': optimalConfig_disc['model.transformer.embed_dim'],
        'transformer_depth': optimalConfig_disc['model.transformer.depth'],
        'transformer_heads': optimalConfig_disc['model.transformer.heads'],
        'transformer_mlp_dim': optimalConfig_disc['model.transformer.mlp_dim'],
        'transformer_dropout': optimalConfig_disc['model.transformer.dropout'],
        'use_cls_token': optimalConfig_disc['model.transformer.use_cls_token'],
        'use_pos_embed': optimalConfig_disc['model.transformer.use_positional_encoding'],
        'transformer_stride': optimalConfig_disc['model.transformer.stride'],
        'cls': config['dataLoader.background'],

        # Regression head params
        'regression_hidden_sizes': optimalConfig_disc['model.regression.hidden_sizes'],
        'cls_hidden_sizes': optimalConfig_disc.get('model.cls.hidden_sizes', False),
        'dropout_regression': optimalConfig_disc['model.regression.dropout'],
        'use_multi_head': optimalConfig_disc['model.regression.use_multihead'],
    }
    model_config_disc['cls'] = 0
    model_config_disc['use_multi_head'] = 0
    model_config_disc['discriminator'] = 1

    model_config_disc2 = {
        # Use config['key.with.dots']
        'model_type': optimalConfig_disc2['model.type'],
        'num_bands': config['ae.encoder.channels'][-1],
        'patch_height': 8,
        'patch_width': 8,

        'AE': not config['ae.predictor_input_type'] and config.get('ae.use_encoder_fc', True),
        'conv_dim': config.get('ae.conv_dim', 2),
        # CNN params
        'shared_cnn_channels': optimalConfig_disc2['model.cnn.shared_channels'],
        'shared_cnn_kernels': optimalConfig_disc2['model.cnn.shared_kernels'],
        'shared_cnn_pools': optimalConfig_disc2['model.cnn.shared_pools'],
        'use_batchnorm': optimalConfig_disc2['model.cnn.use_batchnorm'],
        'dropout_shared': optimalConfig_disc2['model.cnn.dropout_shared'],  # Used by CNN and MLP
        'instanceNorm': config['model.instanceNorm'],

        # MLP params
        'shared_mlp_sizes': optimalConfig_disc2['model.mlp.shared_sizes'],

        # Transformer params
        'transformer_patch_size': optimalConfig_disc2['model.transformer.patch_size'],
        'transformer_embed_dim': optimalConfig_disc2['model.transformer.embed_dim'],
        'transformer_depth': optimalConfig_disc2['model.transformer.depth'],
        'transformer_heads': optimalConfig_disc2['model.transformer.heads'],
        'transformer_mlp_dim': optimalConfig_disc2['model.transformer.mlp_dim'],
        'transformer_dropout': optimalConfig_disc2['model.transformer.dropout'],
        'use_cls_token': optimalConfig_disc2['model.transformer.use_cls_token'],
        'use_pos_embed': optimalConfig_disc2['model.transformer.use_positional_encoding'],
        'transformer_stride': optimalConfig_disc2['model.transformer.stride'],
        'cls': config['dataLoader.background'],

        # Regression head params
        'regression_hidden_sizes': optimalConfig_disc2['model.regression.hidden_sizes'],
        'cls_hidden_sizes': optimalConfig_disc2.get('model.cls.hidden_sizes', False),
        'dropout_regression': optimalConfig_disc2['model.regression.dropout'],
        'use_multi_head': optimalConfig_disc2['model.regression.use_multihead'],
    }
    model_config_disc2['cls'] = 0
    model_config_disc2['use_multi_head'] = 0
    model_config_disc2['discriminator'] = 1
    latent_dim = model_config['patch_height'] * model_config['patch_height'] * config['ae.encoder.channels'][-1]

    # --- AE Specific Model Config (add these to your wandb sweep config) ---
    ae_config = {
        'conv_dim': config.get('ae.conv_dim', 2),
        'latent_dim': latent_dim,
        'variational': config.get('ae.variational', False),  # For VAE
        'use_encoder_fc': config.get('ae.use_encoder_fc', True),

        'encoder_channels': config.get('ae.encoder.channels', [32, 64, 128]),
        'encoder_activation': config.get('ae.encoder.activation', 'relu'),
        'encoder_kernels': config.get('ae.encoder.kernels', [3, 1, 1]),
        'encoder_3D_kernel': config.get('ae.encoder.3D_kernel', 5),
        'encoder_strides': config.get('ae.encoder.strides', [1, 1, 1]),
        'encoder_instanceNorm' : config.get('model.instanceNorm',False),

        'decoder_channels': config.get('ae.encoder.channels', [32, 64, 128]),
        'decoder_activation': config.get('ae.decoder.activation', 'relu'),
        'decoder_final_activation': config.get('ae.decoder.final_activation', 'sigmoid'),
        'decoder_kernels': config.get('ae.encoder.kernels', [3, 1, 1]),
        'decoder_3D_kernel': config.get('ae.encoder.3D_kernel', 5),
        'decoder_strides': config.get('ae.encoder.strides', [1, 1, 1]),
        'decoder_instanceNorm' : config.get('model.instanceNorm',False),

        'predictor_hidden_sizes': config.get('ae.predictor.hidden_sizes', [32, 16]),
        'predictor_dropout': config.get('ae.predictor.dropout', 0.2),
        'predictor_activation': config.get('ae.predictor.activation', 'relu'),

        'discriminator_hidden_sizes': config.get('ae.discriminator.hidden_sizes', [64, 32]),
        'discriminator_dropout': config.get('ae.discriminator.dropout', 0.3),
        'discriminator_activation': config.get('ae.discriminator.activation', 'leaky_relu'),
        'cls': config['dataLoader.background'],
        'num_domains': config.get('ae.num_domains', 2)  # dark, morning, afternoon
    }

    # These input dims are needed for Encoder/Decoder init
    # They might vary if you have pre-processing that changes dim before model
    ae_input_channels = 224
    ae_input_height = model_config['patch_height']
    ae_input_width = model_config['patch_width']
    grape_indices = sorted(
            list(set(range(360)) - {318, 319, 320, 321, 322, 323}))
    # ... (total_indexes, testIndexs, trainIndexs logic) ...
    if not config.get('dataLoader.extendRef', False):
        total_indexes = sorted(
            list(set(range(720)) - {318, 319, 320, 321, 322, 323, 678, 679, 680, 681, 682, 683}))
    else:
        total_indexes = sorted(
            list(set(range(1440)) - {318, 319, 320, 321, 322, 323,  678, 679, 680, 681, 682, 683,  1038, 1039,1040,1041,1042,1043, 1398,1399,1400,1401,1402,1403}))
    if config.get('random_test_indexes', True):
        testIndexs = [4, 8, 12, 14, 27, 28, 32, 35, 40, 41, 44, 47, 51, 61, 62, 64, 65, 85,
                      95, 98, 100, 107, 115, 120, 127, 128, 133, 135, 136, 138, 142, 156, 159, 162, 170,
                      171, 178, 186, 197, 200, 206, 213, 215, 216, 221, 224, 226, 230, 240, 242, 254, 256,
                      258, 260, 267, 279, 282, 283, 285, 288, 292, 296, 300, 307, 312, 326, 327, 332, 337,
                      341, 358]

    else:
        testIndexs = [246, 247, 248, 249, 250, 251, 282, 283, 284, 285, 286, 287, 12, 13, 14, 15, 16, 17, 0, 1,
                      2, 3, 4, 5, 162, 163, 164, 165, 166, 167, 78, 79, 80, 81, 82, 83, 288, 289, 290, 291, 292,
                      293, 90, 91, 92, 93, 94, 95, 240, 241, 242, 243, 244, 245, 48, 49, 50, 51, 52, 53, 324,
                      325, 326, 327, 328, 329]
    trainIndexs = sorted(list(set(total_indexes) - set(testIndexs)))
    # --- DataLoaders for AE Robustness ---
    print("Initializing DataLoaders for AE Robustness...")
    common_loader_params = {
        'camera': str(10),
        'front_or_back': 'Front',
        'preprocess_technique': config['dataLoader.processTechnique'],
        'indivdualGrapes': True, 'white_ref': bool(config['dataLoader.whiteRef']),
        'black_ref': bool(config['dataLoader.blackRef']),
        'normalize': config['dataLoader.normalize'],  # Normalization handled by AE decoder target
        'patches': True, 'sg_window': optimalConfig['dataLoader.Savitzky_Golay_window'],
        'patchSize': model_config['patch_height'],  # Use the calculated patch height
        'pixelMean': bool(1),
        'background': True,  # These flags need to map to your datasets
        'probBackground': 0 if not config['dataLoader.background'] else optimalConfig[
            'dataLoader.probBackground'],
        'train': True, 'labo': 1 if not config.get('dataLoader.extendDataJustWhiteRef', False) else 0, 'sweep': False,
        'autoEncoder': True, 'extendedData': [1, 0, 0, 0 if not config.get('dataLoader.dataSet', 0) == 0 else 1], 'extendedDataReferences' : config.get('dataLoader.extendRef', False)
    }

    # Assuming standard train/test split for now, and domain_id comes from loader
    trainDataLoader = grape_dataLoader(
        **common_loader_params, Indexs=trainIndexs, extendDataJustWhiteRef=config.get('dataLoader.extendDataJustWhiteRef', False),
    )
    common_loader_params['scalars'] = [trainDataLoader.scaler_labels_brix,
                                       trainDataLoader.scaler_labels_acid,
                                       trainDataLoader.scaler_images,
                                       trainDataLoader.scaler_images[0],
                                       trainDataLoader.scaler_labels_ratio,
                                       trainDataLoader.scaler_labels_weights]
    testDataLoader = grape_dataLoader(
        **common_loader_params, Indexs=testIndexs, extendDataJustWhiteRef=config.get('dataLoader.extendDataJustWhiteRef', False),   # Use train=False for test
    )
    batch_size = 16
    trainData = Subset(trainDataLoader, trainIndexs)
    testData = Subset(testDataLoader, testIndexs)
    trainloader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=2,
                             drop_last=True, pin_memory=True)
    testloader = DataLoader(testData, batch_size=batch_size, shuffle=False, num_workers=2,
                            drop_last=True, pin_memory=True)

    # --- Model Initialization ---
    encoder = EncoderAE_3D(ae_config).to(device)
    encoder_shape_before_flatten_tuple = encoder.shape_before_flatten
    ae_config['latent_dim'] = encoder.flattened_size
    model_config['encoder_output_shape'] = encoder_shape_before_flatten_tuple
    model_config_disc['encoder_output_shape'] = encoder_shape_before_flatten_tuple
    model_config_disc2['encoder_output_shape'] = encoder_shape_before_flatten_tuple

    if config.get('ae.conv_dim', 2) == 3: model_config['num_bands'] = encoder_shape_before_flatten_tuple[1]
    if config.get('ae.conv_dim', 2) == 3: model_config_disc['num_bands'] = encoder_shape_before_flatten_tuple[1]
    if config.get('ae.conv_dim', 2) == 3: model_config_disc2['num_bands'] = encoder_shape_before_flatten_tuple[1]

    decoder = DecoderAE_3D(ae_config, encoder_shape_before_flatten_tuple).to(device)
    if config['model.simpelOrComplex']:
        sugar_predictor = SpectralPredictorWithTransformer(model_config).to(device)
    else:
        sugar_predictor = SugarPredictorLatent(ae_config).to(device)
    if config['model.simpelOrComplex']:
        domain_discriminator = SpectralPredictorWithTransformer(model_config_disc).to(device)
        domain_discriminator2 = SpectralPredictorWithTransformer(model_config_disc if config.get('ae.discriminator.model.2Same', False) else model_config_disc2).to(device)

    else:
        domain_discriminator = DomainDiscriminator(ae_config).to(device)
        domain_discriminator2 = DomainDiscriminator(ae_config).to(device)

    if config.get('dataLoader.extendRef', False):
        reference_discriminator = DomainDiscriminator(ae_config).to(device)
    # wandb.watch((encoder, decoder, sugar_predictor, domain_discriminator), log_freq=100)

    # --- Optimizers ---
    lr_ae = config.get('optimizer.ae_lr', 1e-3)
    lr_disc = config.get('optimizer.disc_lr', 1e-4)
    optimizer_type = torch.optim.Adam if 'Adam' == config.get('optimizer.type', 'Adam') else torch.optim.AdamW
    optimizer_ae = optimizer_type(
        list(encoder.parameters()) + list(decoder.parameters()) + list(sugar_predictor.parameters()),
        lr=lr_ae,
        weight_decay=config.get('optimizer.ae_weight_decay', 1e-5) if 'Adam' != config.get('optimizer.type',
                                                                                           'Adam') else 0
    )
    optimizer_disc = optimizer_type(
        domain_discriminator.parameters(),
        lr=lr_disc,
        weight_decay=config.get('optimizer.disc_weight_decay', 1e-5) if 'Adam' != config.get('optimizer.type',
                                                                                             'Adam') else 0
    )
    optimizer_disc2 = optimizer_type(
        domain_discriminator2.parameters(),
        lr=lr_disc,
        weight_decay=config.get('optimizer.disc_weight_decay', 1e-5) if 'Adam' != config.get(
            'optimizer.type',
            'Adam') else 0
    )
    # --- Loss Functions ---
    if config['loss.type'] == 'MSELoss':
        criterion_sugar = nn.MSELoss()
    elif config['loss.type'] == 'HuberLoss':
        criterion_sugar = nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss function: {config['loss.type']}")  # Loss function
    criterion_reconstruction = nn.MSELoss()  # or L1Loss
    # criterion_sugar = nn.MSELoss()  # if regression, CrossEntropy if classification
    criterion_domain = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()

    if config['dataLoader.background']:
        reg_LR_brix = optimalConfig['optimizer.reg_LR_brix']
        reg_LR_acid = optimalConfig['optimizer.reg_LR_acid']
    lambda_recon = config.get('loss.lambda_reconstruction', 1.0)
    lambda_sugar = config.get('loss.lambda_sugar', 1.0)  # This is brix prediction
    lambda_adv = config.get('loss.lambda_adversarial', 0.1)
    lambda_kld = config.get('loss.lambda_kld', 0.01)  # For VAE
    lambda_fool = config.get('loss.lambda_fool', 1)
    lambda_discrep = config.get('loss.lambda_discrep', 1)
    lambda_manifold = config.get('loss.lambda_manifold', 1)
    # --- Scheduler ---
    def get_scheduler(optimizer, scheduler_config_key_prefix):
        scheduler_type = config.get(f'{scheduler_config_key_prefix}.type', None)  # e.g. 'ae.scheduler.type'
        if not optimizer or not scheduler_type: return None

        if scheduler_type == 'StepLR':
            return optim.lr_scheduler.StepLR(optimizer,
                                             step_size=config.get(f'{scheduler_config_key_prefix}.step_size',
                                                                  30),
                                             gamma=config.get(f'{scheduler_config_key_prefix}.gamma', 0.1))
        elif scheduler_type == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=config.get(f'{scheduler_config_key_prefix}.T_max',
                                                                         10),
                                                        eta_min=config.get(
                                                            f'{scheduler_config_key_prefix}.eta_min', 0))
        elif scheduler_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode=config.get(
                                                            f'{scheduler_config_key_prefix}.rop_mode',
                                                            'min'),  # Monitor loss for AE/Disc
                                                        factor=config.get(
                                                            f'{scheduler_config_key_prefix}.rop_factor', 0.1),
                                                        patience=config.get(
                                                            f'{scheduler_config_key_prefix}.rop_patience', 10))
        elif scheduler_type == 'ExponentialLR':
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=optimalConfig.get(f'{scheduler_config_key_prefix}.scheduler.gamma', 0.95)
                # Slightly different default maybe
            )
        return None

    scheduler_ae = get_scheduler(optimizer_ae, 'ae.scheduler')
    scheduler_disc = get_scheduler(optimizer_disc, 'disc.scheduler')
    scheduler_disc2 = get_scheduler(optimizer_disc2, 'disc.scheduler')



    disc_train_interval = config.get('ae.discriminator_train_interval', 1)
    # In trainer, during AE model initialization
    grl_layer = None
    if config.get('ae.use_grl', False):
        grl_lambda = config.get('loss.lambda_adversarial', 0.1)  # Reuse lambda_adv
        grl_layer = GradientReversalLayer(lambda_coeff=grl_lambda).to(device)

    # --- Training Loop ---

    best_sugar_r2 = -float('inf')
    best_sugar_mse = float('inf')

    predicted_values_acid = []
    target_values_acid = []
    all_test_pred_brix = []
    all_test_target_brix = []
    all_test_pred_acid = []
    all_test_target_acid = []
    average_test_R2_brix = []
    average_test_R2_acid = []
    epochs = 201
    epoch = 0
    if get_configs: return config, model_config, ae_config, optimalConfig
    while epoch < epochs and epoch < MAX_EPOCHS:
        dataiter = iter(trainloader)
        testiter = iter(testloader)
        encoder.train()
        decoder.train()
        sugar_predictor.train()
        domain_discriminator.train()

        predicted_reg_all_brix = []
        target_reg_all_brix = []
        predicted_reg_all_acid = []
        target_reg_all_acid = []
        target_reg_all_cls = []
        predicted_reg_all_cls = []
        target_disc_all = []
        predicted_disc_all = []

        total_loss_ae_epoch = []
        total_loss_d_epoch = []
        total_loss_recon_epoch = []
        total_loss_sugar_epoch = []
        total_loss_adv_epoch = []
        total_loss_kld_epoch = []  # For VAE

        for batch_idx in range(len(dataiter)):
            data = next(dataiter)
            images = data['image'].to(device).float()
            sugar_labels_true = data['brix'].to(device).float()
            domain_labels_true = data['domain_id'].to(device).long()
            reference_labels_true = data['reference_id'].to(device).long()
            cls_labels_true = data['cls'].to(device).long().squeeze(1)
            acid_labels_true = data['acid'].to(device).float()
            reconstruction_image = data['reconstruction_target'].to(device).float()

            # --- Train Discriminator ---
            if not config.get('ae.evenAndOdd', False) or not epoch%2:
                for _ in range(disc_train_interval):  # Train D k times
                    optimizer_disc.zero_grad()
                    with torch.no_grad():  # Detach encoder output for discriminator training
                        if ae_config['variational'] and ae_config['use_encoder_fc']:
                            latent_mu, _, conv_features = encoder(images)
                            latent_code_disc = latent_mu  # Use mu for discriminator input
                        else:
                            latent_code_disc, conv_features = encoder(images)

                    if config['ae.predictor_input_type']:
                        domain_preds_for_disc = domain_discriminator(conv_features)
                        if config.get('ae.bi_classifier', False):
                            domain_preds_for_disc2 = domain_discriminator2(conv_features)
                        if config.get('dataLoader.extendRef', False): reference_preds_for_disc = reference_discriminator(conv_features)
                    else:
                        domain_preds_for_disc = domain_discriminator(latent_code_disc)
                        if config.get('ae.bi_classifier', False):
                            domain_preds_for_disc2 = domain_discriminator2(latent_code_disc)
                        if config.get('dataLoader.extendRef', False): reference_preds_for_disc = reference_discriminator(latent_code_disc)

                    loss_d = criterion_domain(domain_preds_for_disc, domain_labels_true)
                    if config.get('dataLoader.extendRef', False): loss_d += criterion_domain(reference_preds_for_disc, reference_labels_true)
                    loss_d.backward()
                    if config.get('ae.bi_classifier', False):
                        loss_d2 = criterion_domain(domain_preds_for_disc2, domain_labels_true)
                        loss_d2.backward()
                        optimizer_disc2.step()
                    optimizer_disc.step()
                    scheduler_disc2.step()
                    total_loss_d_epoch.append(loss_d.item())
            if not config.get('ae.evenAndOdd', False) or epoch%2:
                # --- Train AE (Encoder, Decoder, Sugar Predictor) ---
                optimizer_ae.zero_grad()

                # Forward pass for AE
                if ae_config['variational'] and ae_config['use_encoder_fc']:
                    latent_mu, latent_logvar, conv_features = encoder(images)
                    latent_code_ae = reparameterize(latent_mu, latent_logvar)
                    loss_kld = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())
                    loss_kld = loss_kld / images.size(0)  # Average over batch
                else:
                    latent_code_ae, conv_features = encoder(images)
                    loss_kld = torch.tensor(0.0).to(device)  # No KLD for standard AE
                reconstructed_images = decoder(latent_code_ae)

                # Sugar prediction
                if config['ae.predictor_input_type']:
                    sugar_preds = sugar_predictor(conv_features)
                else:
                    sugar_preds = sugar_predictor(latent_code_ae)

                if config['dataLoader.background']:
                    loss_sugar = criterion_cls(sugar_preds[0], cls_labels_true)
                    regression_mask = (cls_labels_true == 1).float()
                    if regression_mask.sum() > 0:
                        output_reg1 = sugar_preds[1][regression_mask == 1]
                        target_reg1 = sugar_labels_true[regression_mask == 1]
                        loss_reg1 = criterion_sugar(output_reg1.squeeze(), target_reg1.squeeze())

                        output_reg2 = sugar_preds[2][regression_mask == 1]
                        target_reg2 = acid_labels_true[regression_mask == 1]
                        loss_reg2 = criterion_sugar(output_reg2.squeeze(), target_reg2.squeeze())
                        loss_sugar += reg_LR_brix * loss_reg1 + reg_LR_acid * loss_reg2
                else:
                    loss_sugar = criterion_sugar(sugar_preds.squeeze(1), sugar_labels_true.squeeze())

                # AE tries to fool discriminator
                # Adversarial part
                if grl_layer:
                    features_for_discriminator_adv = grl_layer(latent_code_ae if not config['ae.predictor_input_type'] else conv_features)  # Apply GRL
                else:
                    features_for_discriminator_adv = latent_code_ae if not config['ae.predictor_input_type'] else conv_features

                if config['ae.predictor_input_type']:
                    domain_preds_for_ae = domain_discriminator(features_for_discriminator_adv)
                    if config.get('ae.bi_classifier', False):
                        domain_preds_ae_d2 = domain_discriminator2(features_for_discriminator_adv)

                    if config.get('dataLoader.extendRef', False): reference_preds_for_disc = reference_discriminator(features_for_discriminator_adv)

                else:
                    domain_preds_for_ae = domain_discriminator(features_for_discriminator_adv)
                    if config.get('ae.bi_classifier', False):
                        domain_preds_ae_d2 = domain_discriminator2(features_for_discriminator_adv)

                    if config.get('dataLoader.extendRef', False): reference_preds_for_disc = reference_discriminator(features_for_discriminator_adv)


                # domain_preds_for_ae = domain_discriminator(features_for_discriminator_adv)
                loss_adv_d2 = torch.tensor(0.0).to(device)
                if grl_layer:
                    # With GRL, the objective for encoder is to make discriminator output *correctly* for true labels
                    # because the GRL will flip the gradient direction for the encoder.
                    loss_adv = criterion_domain(domain_preds_for_ae, domain_labels_true)
                    if config.get('ae.bi_classifier', False):
                        loss_adv_d2 = criterion_domain(domain_preds_ae_d2, domain_labels_true)
                    if config.get('dataLoader.extendRef', False): loss_adv += criterion_domain(reference_preds_for_disc, reference_labels_true)

                else:
                    # Original adversarial loss (e.g., negative CE loss)
                    flipped_domain_labels = 1 - domain_labels_true
                    flipped_reference_labels = 1 - reference_labels_true
                    loss_adv = criterion_domain(domain_preds_for_ae, flipped_domain_labels)
                    if config.get('ae.bi_classifier', False):
                        loss_adv_d2 = criterion_domain(domain_preds_ae_d2, flipped_domain_labels)
                    if config.get('dataLoader.extendRef', False): loss_adv += criterion_domain(reference_preds_for_disc, flipped_reference_labels)
                if config.get('ae.bi_classifier', False):
                    field_mask = (domain_labels_true == 1)  # Assuming 1 is field
                    if field_mask.sum() > 0:  # Only if there are field samples in the batch
                        if ae_config['variational'] and ae_config['use_encoder_fc']:
                            actual_features_for_discrepancy = (latent_code_ae[field_mask]
                                                               if not config['ae.predictor_input_type']
                                                               else conv_features[field_mask])
                        else:
                            actual_features_for_discrepancy = (latent_code_ae[field_mask]
                                                               if not config['ae.predictor_input_type']
                                                               else conv_features[field_mask])

                        preds_d1_target_raw = domain_discriminator(actual_features_for_discrepancy)
                        preds_d2_target_raw = domain_discriminator2(actual_features_for_discrepancy)

                        # Example: L1 distance for discrepancy (can also use L2, or JS/KL on softmax outputs)
                        loss_discrepancy = torch.mean(torch.abs(preds_d1_target_raw - preds_d2_target_raw))
                    else:
                        loss_discrepancy = torch.tensor(0.0).to(device)

                    loss_adv = lambda_fool * (loss_adv + loss_adv_d2) + lambda_discrep * loss_discrepancy
                # Losses for AE
                if not config['ae.normalizedReconstruction']:
                    loss_recon = criterion_reconstruction(reconstructed_images, images)
                else:
                    loss_recon = criterion_reconstruction(reconstructed_images, reconstruction_image)
                # loss_adv = -criterion_domain(domain_preds_for_ae, domain_labels_true)\
                loss_manifold_reg = torch.tensor(0.0).to(device)
                if config.get('ae.manifold_reg', False):
                    random_grapes_lab_indices = random.choices(grape_indices, k=8)  # Sample 8 LAB grape indices
                    random_grapes_field_indices = [i + 360 for i in random_grapes_lab_indices]

                    grapes_lab_samples = []
                    grapes_field_samples = []

                    for lab_idx, field_idx in zip(random_grapes_lab_indices, random_grapes_field_indices):
                        # Assuming trainDataLoader is the torch.utils.data.Dataset instance
                        # and it returns a dictionary {'image': ..., 'brix': ...}
                        grapes_lab_samples.append(
                            trainDataLoader[lab_idx]['image'].unsqueeze(0))  # Get 'image' tensor
                        grapes_field_samples.append(
                            trainDataLoader[field_idx]['image'].unsqueeze(0))  # Get 'image' tensor

                    grapes_lab_batch = torch.cat(grapes_lab_samples).to(device).float()
                    grapes_field_batch = torch.cat(grapes_field_samples).to(device).float()

                    # 5. Pass these new mini-batches through the *current* encoder
                    #    This happens INSIDE the main training loop for the current batch
                    if ae_config['variational'] and ae_config['use_encoder_fc']:
                        mu_lab_pair, _, _ = encoder(
                            grapes_lab_batch)  # Assuming encoder returns (mu, logvar, conv_features)
                        mu_field_pair, _, _ = encoder(grapes_field_batch)
                    elif ae_config['variational']:  # VAE but no FC encoder output
                        mu_lab_pair, _ = encoder(grapes_lab_batch)  # Assuming encoder returns (mu, logvar)
                        mu_field_pair, _ = encoder(grapes_field_batch)
                    else:  # Standard AE
                        # If not VAE, you'd use the direct latent code. Let's assume it's the first output.
                        # This part needs to align with how your non-VAE encoder returns latent codes.
                        # For simplicity, I'll assume it's the first element like VAE's mu for now.
                        latent_lab_pair, _ = encoder(grapes_lab_batch)  # Placeholder, adjust if non-VAE
                        latent_field_pair, _ = encoder(grapes_field_batch)  # Placeholder, adjust if non-VAE
                        mu_lab_pair = latent_lab_pair  # Use direct latent code for distance
                        mu_field_pair = latent_field_pair  # Use direct latent code for distance

                    # 6. Calculate L2 distance between the means (or latent codes)
                    #    The sum here is over all dimensions of the latent codes and then implicitly summed over the 8 pairs
                    #    if you add it directly. Or, mean over pairs.
                    distance_per_pair = torch.sum((mu_lab_pair - mu_field_pair).pow(2),
                                                  dim=1)  # Sum over latent dims, keep batch dim
                    loss_manifold_reg = torch.mean(distance_per_pair)  # Average distance over the 8 pairs

                    # This loss_manifold_reg will be added to total_loss_ae

                if config.get('ae.no_reconstruction',False): lambda_recon = 0
                total_loss_ae = (0 * loss_recon + #lambda_recon
                                 lambda_sugar * loss_sugar +
                                 0 * loss_adv + #lambda_adv
                                 0 * loss_manifold_reg +  # New manifold loss  lambda_manifold
                                 (lambda_kld * loss_kld if ae_config['variational'] else 0.0)
                                 )
                total_loss_ae.backward()
                optimizer_ae.step()

                total_loss_ae_epoch.append(total_loss_ae.item())
                total_loss_recon_epoch.append(loss_recon.item())
                total_loss_sugar_epoch.append(loss_sugar.item())
                total_loss_adv_epoch.append(loss_adv.item())
                if ae_config['variational']:
                    total_loss_kld_epoch.append(loss_kld.item())

        # --- AE Evaluation ---
        encoder.eval()
        decoder.eval()
        sugar_predictor.eval()
        domain_discriminator.eval()

        all_sugar_preds_test = []
        all_sugar_targets_test = []
        all_domain_preds_test = []
        all_domain_targets_test = []
        total_test_loss_sugar_epoch = []

        with torch.no_grad():
            for batch_idx in range(len(testiter)):
                data = next(testiter)
                images = data['image'].to(device).float()
                sugar_labels_true = data['brix'].to(device).float()
                domain_labels_true = data['domain_id'].to(device).long()
                cls_labels_true = data['cls'].to(device).long().squeeze(1)
                acid_labels_true = data['acid'].to(device).float()

                if ae_config['variational'] and ae_config['use_encoder_fc']:
                    latent_mu, _, conv_features = encoder(images)  # Use mu for eval
                    latent_code_eval = latent_mu
                else:
                    latent_code_eval, conv_features = encoder(images)

                target_disc_all.append(domain_labels_true)
                target_reg_all_cls.append(cls_labels_true)

                # Sugar prediction
                if config['ae.predictor_input_type']:
                    sugar_preds = sugar_predictor(conv_features)
                else:
                    sugar_preds = sugar_predictor(latent_code_eval)

                if config['dataLoader.background']:
                    predicted_reg_all_cls.append(sugar_preds[0].squeeze(1))
                    regression_mask = (cls_labels_true == 1).float()
                    if regression_mask.sum() > 0:
                        output_reg1 = sugar_preds[1][regression_mask == 1]
                        target_reg1 = sugar_labels_true[regression_mask == 1]

                        output_reg2 = sugar_preds[2][regression_mask == 1]
                        target_reg2 = acid_labels_true[regression_mask == 1]

                        target_reg_all_brix.append(target_reg1)
                        target_reg_all_acid.append(target_reg2)
                        predicted_reg_all_brix.append(output_reg1)
                        predicted_reg_all_acid.append(output_reg2)
                        sugar_labels_true = target_reg1
                else:
                    output_reg1 = sugar_preds.squeeze()
                    predicted_reg_all_brix.append(output_reg1)
                    target_reg_all_brix.append(sugar_labels_true)

                sugar_preds_test = output_reg1
                if config['ae.predictor_input_type']:
                    domain_preds_test = domain_discriminator(conv_features)
                else:
                    domain_preds_test = domain_discriminator(latent_code_disc)
                # domain_preds_test = domain_discriminator(latent_code_eval)

                predicted_disc_all.append(domain_preds_test.squeeze(1))
                if not config['dataLoader.background']:
                    loss_sugar_test = criterion_sugar(sugar_preds_test.squeeze(), sugar_labels_true.squeeze())
                elif regression_mask.sum() > 0:
                    loss_sugar_test = criterion_sugar(sugar_preds_test.squeeze(), sugar_labels_true.squeeze())

                # loss_sugar_test = criterion_sugar(sugar_preds_test.squeeze(), sugar_labels_true.squeeze())

                total_test_loss_sugar_epoch.append(loss_sugar_test.item())

                all_sugar_preds_test.append(sugar_preds_test.cpu())
                all_sugar_targets_test.append(sugar_labels_true.cpu())
                all_domain_preds_test.append(domain_preds_test.argmax(dim=1).cpu())
                all_domain_targets_test.append(domain_labels_true.cpu())
        predicted_value_brix = torch.cat(predicted_reg_all_brix, dim=0).cpu().detach()
        target_reg_brix = torch.cat(target_reg_all_brix, dim=0).cpu().detach()
        target_value_brix = target_reg_brix.squeeze()
        r2_sugar_test, mse_sugar_test = get_metrics(predicted_value_brix.squeeze(), target_value_brix)
        average_test_R2_brix.append(r2_sugar_test)
        predicted_disc = torch.cat(predicted_disc_all, dim=0).cpu().detach()
        target_disc = torch.cat(target_disc_all, dim=0).cpu().detach()
        correct_predictions_disc = (predicted_disc.argmax(dim=1) == target_disc).sum().item()
        total_predictions_disc = target_disc.size(0)
        classification_accuracy_disc = correct_predictions_disc / total_predictions_disc
        if config['dataLoader.background']:
            predicted_cls = torch.cat(predicted_reg_all_cls, dim=0).cpu().detach()
            target_cls = torch.cat(target_reg_all_cls, dim=0).cpu().detach()

            predicted_value_acid = torch.cat(predicted_reg_all_acid, dim=0).cpu().detach()
            target_reg = torch.cat(target_reg_all_acid, dim=0).cpu().detach()
            target_value_acid = target_reg.squeeze()
            r_two_acid, mse_acid = get_metrics(predicted_value_acid.squeeze(), target_value_acid)
            average_test_R2_acid.append(r_two_acid)
            correct_predictions = (predicted_cls.argmax(dim=1) == target_cls).sum().item()
            total_predictions = target_cls.size(0)
            classification_accuracy = correct_predictions / total_predictions

        # --- Scheduler Step ---
        disc_acc = (torch.cat(all_domain_preds_test, dim=0).squeeze() == torch.cat(all_domain_targets_test,
                                                                                   dim=0).squeeze()).float().mean().item()
        avg_test_loss_s = np.mean(total_test_loss_sugar_epoch) if total_test_loss_sugar_epoch else 0
        if scheduler_ae:
            if isinstance(scheduler_ae, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_ae.step(avg_test_loss_s)  # or r2_s
            else:
                scheduler_ae.step()
        if scheduler_disc:
            if isinstance(scheduler_disc, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_disc.step(disc_acc)  # Monitor disc accuracy
            else:
                scheduler_disc.step()

        if epoch == 0:
            predicted_values_brix = predicted_value_brix
            target_values_brix = target_value_brix
            if config['dataLoader.background']:
                predicted_values_acid = predicted_value_acid
                target_values_acid = target_value_acid
        if r2_sugar_test > best_sugar_r2:
            best_sugar_r2 = r2_sugar_test
            predicted_values_brix = predicted_value_brix
            target_values_brix = target_value_brix
            if config['dataLoader.background']:
                predicted_values_acid = predicted_value_acid
                target_values_acid = target_value_acid
        if mse_sugar_test < best_sugar_mse:
            best_sugar_mse = mse_sugar_test
        if best_sugar_r2 < -0.2 and epoch == 5:
            wandb.log({'to_low_R2': True})
            break
        # elif best_sugar_r2 >= -0.2 and epoch == 5:
        #     potential_run = True
        if epoch in model_epoch_save:
            transfer_brix = transfer_loop(common_loader_params.copy(), encoder,sugar_predictor,ae_config,config,wandb,epoch, config.get('dataLoader.dataSet', 2))
            # models.append([copy.deepcopy(encoder), copy.deepcopy(sugar_predictor)])
            if (transfer_brix > 0.2 and 50 <= epoch < 101 and not epochs == 201) or (transfer_brix > 0.25 and 101 <= epoch < 201 and not epochs == 301) or (transfer_brix > 0.25 and 201 <= epoch < 301 and not epochs == 401):
                if not epochs > 400:
                    epochs += 100
            if transfer_brix > 0.30:
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'sugar_predictor_state_dict': sugar_predictor.state_dict(),
                }, f'models/ae_predictor_epoch_{epoch}_R2_{transfer_brix}_runID_{run_id}.pth')
        wandb.log({
            'epoch': epoch,
            'train_loss_ae': np.mean(total_loss_ae_epoch) if total_loss_ae_epoch else 0,
            'train_loss_discriminator': np.mean(total_loss_d_epoch) if total_loss_d_epoch else 0,
            'train_loss_reconstruction': np.mean(total_loss_recon_epoch) if total_loss_recon_epoch else 0,
            'train_loss_sugar': np.mean(total_loss_sugar_epoch) if total_loss_sugar_epoch else 0,
            'train_loss_adversarial': np.mean(total_loss_adv_epoch) if total_loss_adv_epoch else 0,
            'train_loss_kld': np.mean(total_loss_kld_epoch) if total_loss_kld_epoch and ae_config[
                'variational'] else 0,
            'test_r2_brix': r2_sugar_test,
            'test_mse_brix': mse_sugar_test,
            'test_avg_loss_sugar': np.mean(
                total_test_loss_sugar_epoch) if total_test_loss_sugar_epoch else 0,
            'test_discriminator_accuracy': disc_acc,
            'classification_accuracy': 0 if not config['dataLoader.background'] else classification_accuracy,
            'r_two_acid': 0 if not config['dataLoader.background'] else r_two_acid,
            'mse_acid': 0 if not config['dataLoader.background'] else mse_acid,
            'discriminator_accuracy': classification_accuracy_disc,
            # 'learning_rate_ae': optimizer_ae.param_groups[0]['lr'], # If using scheduler
            # 'learning_rate_disc': optimizer_disc.param_groups[0]['lr'],
        })
        epoch += 1

    # --- Final Logging and Unnormalization (for existing model type) ---
    normalizer_brix = trainDataLoader.scaler_labels_brix
    normalizer_acid = trainDataLoader.scaler_labels_acid
    if config['dataLoader.normalize']:
        predicted_values_brix = normalizer_brix.inverse_transform(predicted_values_brix.reshape(-1, 1))
        target_values_brix = normalizer_brix.inverse_transform(target_values_brix.reshape(-1, 1))
        if config['dataLoader.background']:
            predicted_values_acid = normalizer_acid.inverse_transform(predicted_values_acid.reshape(-1, 1))
            target_values_acid = normalizer_acid.inverse_transform(target_values_acid.reshape(-1, 1))
    all_test_pred_brix = np.concatenate((all_test_pred_brix, predicted_values_brix.squeeze()))
    all_test_target_brix = np.concatenate((all_test_target_brix, target_values_brix.squeeze()))
    if config['dataLoader.background']:
        all_test_pred_acid = np.concatenate((all_test_pred_acid, predicted_values_acid.squeeze()))
        all_test_target_acid = np.concatenate((all_test_target_acid, target_values_acid.squeeze()))

    r_two_brix, mse_brix = get_metrics(torch.tensor(all_test_pred_brix), torch.tensor(all_test_target_brix))
    if config['dataLoader.background']:
        r_two_acid, mse_acid = get_metrics(torch.tensor(all_test_pred_acid), torch.tensor(all_test_target_acid))
    wandb.log({
        'min_mse_brix': mse_brix,
        'devation_brix': np.sqrt(mse_brix),
        'max_r2_brix': r_two_brix,
        'min_mse_acid': 0 if not config['dataLoader.background'] else mse_acid,
        'max_r2_acid': 0 if not config['dataLoader.background'] else r_two_acid,
        'devation_acid': 0 if not config['dataLoader.background'] else np.sqrt(mse_acid),
        'average_test_R2_brix': np.mean(average_test_R2_brix[-10:]),
        'average_test_R2_acid': 0 if not config['dataLoader.background'] else np.mean(
            average_test_R2_acid[-10:]),
    })

def create_final_map(image_path, results_list, output_path="results/final_annotated_map.png", saveFig = True):
    """
    Overlays the analysis results onto the full RGB image of the vineyard row.

    Args:
        image_path (str): The file path to the full RGB image (e.g., '3.png').
        results_list (list): The list of dictionaries, where each dictionary
                             contains the analysis for one grape bunch, including
                             the 'global_bounding_box'.
        output_path (str): The path to save the final annotated image.
    """
    if isinstance(image_path, str):
        try:
            # 1. Load the background image
            img = Image.open(image_path)
            img_width, img_height = img.size
            print(f"Successfully loaded image '{image_path}' with dimensions {img.size}.")
        except FileNotFoundError:
            print(f"Error: The image file was not found at '{image_path}'.")
            print("Please check the path and try again.")
            return
    else:
        img = image_path
        img_width, img_height = img.size
    # 2. Set up the plot using Matplotlib
    # Using a figure size that respects the image's aspect ratio
    fig, ax = plt.subplots(1, figsize=(20, 20 * (img_height / img_width)))
    ax.imshow(img)
    line_width = 0.5  # Thinner line width for the box
    font_size = 0.5  # Smaller font size for the label
    # 3. Iterate through results and draw bounding boxes and text
    for result in results_list:
        bbox = result.get('global_bounding_box')
        if not bbox:
            print(f"Warning: Skipping result for bunch_id {result.get('bunch_id')} due to missing bounding box.")
            continue

        b_id = result.get('bunch_id', 'N/A')
        brix = result.get('predicted_brix', 0)
        acid = result.get('predicted_acidity', 0)
        weight = result.get('predicted_weight_g', 0)

        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1

        # Create a Rectangle patch
        # The color can be changed. 'edgecolor' for the border, 'facecolor' for the fill.
        rect = patches.Rectangle(
            (x1, y1),
            box_width,
            box_height,
            linewidth=line_width,
            edgecolor='lime',  # A bright green color for good visibility
            facecolor='none'   # No fill
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

        # 4. Add the annotation text above the box
        label_text = f"ID:{b_id} | Brix:{brix:.1f} | Acid:{acid:.1f} | W:{weight:.0f}g"

        ax.text(
            x1, y1 - 10,  # Position the text slightly above the top-left corner
            label_text,
            color='white',
            fontsize=font_size,
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
        )

    # 5. Clean up the plot and save it
    ax.axis('off')  # Hide the axes ticks and labels
    plt.tight_layout(pad=0)

    # Save the figure
    if saveFig:
        plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.show()
    print(f"Final annotated map saved to '{output_path}'.")


def create_final_map_pdf(image_path, results_list, output_path="final_annotated_map.png"):
    """
    Overlays analysis results onto the full RGB image and saves it as a
    high-quality, zoomable PDF with smaller annotations.

    Args:
        image_path (str): The file path to the full RGB image.
        results_list (list): The list of dictionaries containing analysis results.
        output_path (str): The path to save the final annotated PDF.
    """
    try:
        # 1. Load the background image using Pillow
        img = Image.open(image_path)
        img_width_px, img_height_px = img.size
        print(f"Successfully loaded image '{image_path}' with dimensions {img.size} pixels.")
    except FileNotFoundError:
        print(f"Error: The image file was not found at '{image_path}'.")
        return
    except Exception as e:
        print(f"An error occurred while opening the image: {e}")
        return

    # --- Set up figure for 1:1 pixel mapping ---
    DPI = 100  # This is a reference for sizing the rasterized image part
    fig_width_in = img_width_px / DPI
    fig_height_in = img_height_px / DPI

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=DPI)
    ax.imshow(img)

    # 3. Iterate through results and draw annotations
    for result in results_list:
        bbox = result.get('global_bounding_box')
        if not bbox:
            continue

        # --- MODIFIED PARAMETERS ---
        line_width = 0.02  # Thinner line width for the box
        font_size = 0.2  # Smaller font size for the label

        b_id = result.get('bunch_id', 'N/A')
        brix = result.get('predicted_brix', 0)
        acid = result.get('predicted_acidity', 0)
        weight = result.get('predicted_weight_g', 0)

        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle(
            (x1, y1), (x2 - x1), (y2 - y1),
            linewidth=line_width,  # Use the new thinner width
            edgecolor='lime',
            facecolor='none'
        )
        ax.add_patch(rect)

        label_text = f"ID:{b_id} | Brix:{brix:.1f} | Acid:{acid:.1f} | W:{weight:.0f}g"
        ax.text(
            x1, y1 - 5,  # Adjust vertical position slightly for smaller font
            label_text,
            color='white',
            fontsize=font_size,  # Use the new smaller font size
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
        )

    # 4. Configure axes for a clean, borderless output
    ax.axis('off')
    plt.tight_layout(pad=0)

    # 5. Save the figure as a PDF
    # The image part will be rasterized, but the text and rectangles will be vectors.
    plt.savefig(
        output_path,
        format='png',  # Explicitly set the format to PDF
        dpi=300,  # High DPI for the rasterized image part
        bbox_inches='tight',
        pad_inches=0
    )

    print(f"Final annotated map saved as a high-quality PDF to '{output_path}'.")

    # It's good practice to close the figure to free up memory
    plt.close(fig)


def create_final_map_png(image_input, results_list, output_path="results/final_annotated_map.png"):
    """
    Overlays analysis results onto an image, provided as a path or an array.

    Args:
        image_input (str, np.ndarray, or PIL.Image.Image):
            The background image. Can be one of:
            - A string file path (e.g., 'vineyard.png').
            - A NumPy array (can be float or uint8, will be converted).
            - A Pillow Image object.
        results_list (list): A list of dictionaries, where each contains
                             the analysis for one grape bunch, including
                             the 'global_bounding_box'.
        output_path (str): The path to save the final annotated image.
    """
    # 1. Load or prepare the background image based on its type
    img = None
    if isinstance(image_input, str):
        try:
            img = Image.open(image_input)
            print(f"Successfully loaded image from path '{image_input}'.")
        except FileNotFoundError:
            print(f"Error: The image file was not found at '{image_input}'.")
            return

    # --- MODIFIED BLOCK START ---
    elif isinstance(image_input, np.ndarray):
        print("Input is a NumPy array. Converting to a Pillow Image.")

        # Pillow's fromarray expects a uint8 data type (0-255).
        # If the input array is float, we must convert it correctly.
        if image_input.dtype != np.uint8:
            print(f"NumPy array is of type '{image_input.dtype}', converting to 'uint8'.")

            # If the data range is 0.0 to 1.0, we can just multiply by 255.
            if image_input.max() <= 1.0:
                image_input = (image_input * 255).astype(np.uint8)
            # Otherwise, perform a full min-max normalization to be safe.
            else:
                img_norm = (image_input - image_input.min()) / (image_input.max() - image_input.min())
                image_input = (img_norm * 255).astype(np.uint8)

        img = Image.fromarray(image_input)
    # --- MODIFIED BLOCK END ---

    elif isinstance(image_input, Image.Image):
        print("Input is a Pillow Image object. Using it directly.")
        img = image_input
    else:
        print(f"Error: Unsupported input type for 'image_input': {type(image_input)}.")
        print("Please provide a file path (str), a NumPy array, or a Pillow Image object.")
        return

    img_width, img_height = img.size
    print(f"Processing image with dimensions {img_width}x{img_height}.")

    # 2. Set up the plot using Matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 12 * (img_height / img_width)))
    ax.imshow(img)
    line_width = 0.5
    font_size = 1

    # 3. Iterate through results and draw bounding boxes and text
    for result in results_list:
        bbox = result.get('global_bounding_box')
        if not bbox:
            print(f"Warning: Skipping result for bunch_id {result.get('bunch_id')} due to missing bounding box.")
            continue

        b_id = result.get('bunch_id', 'N/A')
        brix = result.get('predicted_brix', 0)
        acid = result.get('predicted_acidity', 0)
        weight = result.get('predicted_weight_g', 0)

        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), box_width, box_height,
            linewidth=line_width,
            edgecolor='lime',
            facecolor='none'
        )
        ax.add_patch(rect)

        label_text = f"ID:{b_id} | Brix:{brix:.1f} | Acid:{acid:.1f} | W:{weight:.0f}g"
        ax.text(
            x1, y1 - 10,
            label_text,
            color='white',
            fontsize=font_size,
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
        )

    # 5. Clean up the plot and save it
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    print(f"Final annotated map saved to '{output_path}'.")
def create_final_map_png_2(image_input, results_list, output_path="results/final_annotated_map.png", saveFig=True, showFig=False):
    """
    Overlays analysis results onto an image and saves it at a high resolution.

    Args:
        image_input (str, np.ndarray, or PIL.Image.Image):
            The background image.
        results_list (list): List of dictionaries containing analysis results.
        output_path (str): The path to save the final annotated image.
    """
    # --- IMAGE PREPARATION (unchanged) ---
    img = None
    if isinstance(image_input, str):
        try:
            img = Image.open(image_input)
            print(f"Successfully loaded image from path '{image_input}'.")
        except FileNotFoundError:
            print(f"Error: The image file was not found at '{image_input}'.")
            return
    elif isinstance(image_input, np.ndarray):
        print("Input is a NumPy array. Converting to a Pillow Image.")
        if image_input.dtype != np.uint8:
            print(f"NumPy array is of type '{image_input.dtype}', converting to 'uint8'.")
            if image_input.max() <= 1.0 and image_input.min() >= 0.0:
                 image_input = (image_input * 255).astype(np.uint8)
            else:
                 img_norm = (image_input - image_input.min()) / (image_input.max() - image_input.min())
                 image_input = (img_norm * 255).astype(np.uint8)
        img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        print("Input is a Pillow Image object. Using it directly.")
        img = image_input
    else:
        print(f"Error: Unsupported input type for 'image_input': {type(image_input)}.")
        return

    img_width, img_height = img.size
    print(f"Processing image with dimensions {img_width}x{img_height}.")

    # --- HIGH-RESOLUTION PLOTTING SETUP ---

    # 1. Define the desired output resolution in Dots Per Inch.
    #    300 is great for high-quality images. Use 600 for print quality.
    DPI = 300

    # 2. Calculate the figure size in inches to match the image pixel dimensions 1:1.
    #    This is the key to preventing Matplotlib from resizing your background image.
    fig_width_in = img_width / DPI
    fig_height_in = img_height / DPI

    # 3. Create the figure with the calculated size.
    fig, ax = plt.subplots(1, figsize=(fig_width_in, fig_height_in), dpi=DPI)
    ax.imshow(img)

    # 4. Make annotation sizes (lines, fonts) dynamic and proportional to image size.
    #    This ensures they look good on both small and very large images.
    line_width = int(img_width / 4000)       # e.g., line width of 2 for an 8000px image
    font_size =  int(img_width / 400) #5      # e.g., font size 40 for an 8000px image
    text_offset = int(img_width / 4000)   # Adjust vertical text position

    print(f"Using dynamic annotation sizes: line_width={line_width}, font_size={font_size}")

    # --- ANNOTATION LOOP (unchanged logic, but uses dynamic sizes now) ---
    for result in results_list:
        bbox = result.get('global_bounding_box')
        if not bbox:
            print(f"Warning: Skipping result for bunch_id {result.get('bunch_id')} due to missing bounding box.")
            continue

        b_id = result.get('bunch_id', 'N/A')
        brix = result.get('predicted_brix', 0)
        acid = result.get('predicted_acidity', 0)
        weight = result.get('predicted_weight_g', 0)

        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), box_width, box_height,
            linewidth=line_width, # Use dynamic line width
            edgecolor='lime',
            facecolor='none'
        )
        ax.add_patch(rect)

        label_text = f"ID:{b_id} | Brix:{brix:.1f} | Acid:{acid:.1f} | W:{weight:.0f}g"
        ax.text(
            x1, y1 - text_offset, # Use dynamic text offset
            label_text,
            color='white',
            fontsize=font_size,   # Use dynamic font size
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
        )

    # --- SAVE THE HIGH-RESOLUTION FIGURE ---
    ax.axis('off')
    plt.tight_layout(pad=0)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the figure using the same DPI we used to create it.
    if saveFig:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
        print(f"High-resolution annotated map saved to '{output_path}'.")
    if showFig:
        plt.show()
    plt.close(fig) # Close the figure to free up memory


def plot_predictions(all_test_pred, all_test_target, wandb):
    run = wandb.run
    table = wandb.Table(columns=["Predicted", "Target"])
    for pred, targ in zip(all_test_pred, all_test_target):
        table.add_data(pred.item(), targ.item())
    artifact = wandb.Artifact(f"epoch_data", type="dataset")
    artifact.add(table, "predictions_table")
    run.log_artifact(artifact)

    time.sleep(3)
    api = wandb.Api()
    # Access the specific run
    run_art = api.run(f"{run.entity}/{run.project}/{run.id}")
    latest_artifact = max(run_art.logged_artifacts(), key=lambda artifact: artifact.updated_at)
    table = latest_artifact.get("predictions_table")
    pred_values = np.array([row[0] for row in table.data])
    targ_values = np.array([row[1] for row in table.data])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=targ_values,
            y=pred_values,
            mode='markers',
            name='Predictions vs Targets'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min(targ_values), max(targ_values)],
            y=[min(targ_values), max(targ_values)],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        )
    )
    fig.update_layout(
        title='Predicted vs Target Values',
        xaxis_title='Target Values',
        yaxis_title='Predicted Values',
        showlegend=True,
        template='plotly_white'
    )
    run.log({"Predictions vs Targets": fig})
    # img_bytes = fig.to_image(format="png")
    # img = Image.open(io.BytesIO(img_bytes))
    # img_array = np.array(img)
    # run.log({"Predictions vs Targets image": wandb.Image(img_array)})


def plot_predictions2(all_test_pred, all_test_target, wandb_run, label ='brix'):
    """
    Logs a prediction table artifact and a scatter plot of predictions vs targets.

    Args:
        all_test_pred (list or torch.Tensor): A list or tensor of predicted values.
        all_test_target (list or torch.Tensor): A list or tensor of target values.
        wandb_run: The active wandb run object (e.g., from wandb.init()).
    """
    # ---- 1. Log the Table as an Artifact (This part is fine) ----
    table = wandb.Table(columns=["Predicted", "Target"])

    # Extract numerical values from tensors if necessary
    # .item() is for single-element tensors. If they are multi-element, you might need to iterate differently.
    # Assuming all_test_pred and all_test_target are lists of tensors.
    pred_values = [p.item() for p in all_test_pred]
    targ_values = [t.item() for t in all_test_target]

    for pred, targ in zip(pred_values, targ_values):
        table.add_data(pred, targ)

    artifact = wandb.Artifact(f"predictions", type="predictions")
    artifact.add(table, "predictions_table")
    wandb_run.log_artifact(artifact)

    # ---- 2. Create the Plot Using Local Data (The Efficient Way) ----
    # No need to sleep, use the API, or download anything.
    # We already have the data in pred_values and targ_values.

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=targ_values,
            y=pred_values,
            mode='markers',
            name='Predictions vs Targets'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min(targ_values), max(targ_values)],
            y=[min(targ_values), max(targ_values)],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        )
    )
    fig.update_layout(
        title='Predicted vs Target Values',
        xaxis_title='Target Values',
        yaxis_title='Predicted Values',
        showlegend=True,
        template='plotly_white'
    )

    # Log the figure directly to the run
    wandb_run.log({f"Predictions vs Targets {label}": fig})


def get_best_metric_from_sweep(wandb, metric_name):
    """
    Queries the W&B API to find the best value for a given metric in a sweep.

    Args:
        sweep_id (str): The ID of the W&B sweep.
        metric_name (str): The name of the metric to find the best value for (e.g., 'cross_val_R2').

    Returns:
        float: The best metric value found so far in the sweep.
               Returns -infinity if no runs are found or the metric is not present.
    """
    # If not running in a sweep, there's no "best" to compare to.
    # Return a very low number so the first run is always considered the best.

    api = wandb.Api()
    # The full path to the sweep is needed
    sweep_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.sweep_id}"

    try:
        sweep = api.sweep(sweep_path)
    except wandb.errors.CommError:
        print(f"Could not find sweep at path: {sweep_path}. This might happen on the first run of a new sweep.",
              file=sys.stderr)
        # Fallback to treating this as the first/best run
        return -float('inf')

    best_metric_value = -float('inf')

    # Iterate over all runs in the sweep
    for run in sweep.runs:
        # Check if the run has finished and has the metric in its summary
        if run.state == "finished" and metric_name in run.summary:
            run_metric_value = run.summary[metric_name]
            if isinstance(run_metric_value, str):
                run_metric_value = -float('inf')
            if run_metric_value > best_metric_value:
                best_metric_value = run_metric_value

    return best_metric_value


def mmd_rbf(X, Y, gamma=1.0):
    """
    Calculates the Maximum Mean Discrepancy (MMD) between two sets of samples, X and Y.
    """
    # Ensure X and Y are numpy arrays before converting to tensor
    X = torch.from_numpy(np.asarray(X)).float()
    Y = torch.from_numpy(np.asarray(Y)).float()

    XX = torch.cdist(X, X)
    YY = torch.cdist(Y, Y)
    XY = torch.cdist(X, Y)

    K_XX = torch.exp(-gamma * XX**2).mean()
    K_YY = torch.exp(-gamma * YY**2).mean()
    K_XY = torch.exp(-gamma * XY**2).mean()

    return K_XX + K_YY - 2 * K_XY

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import LabelEncoder
def get_MMD(domain_labels, flattend_all_patches, flattend_all_patchesAE, num_lab_samples, num_am_samples):
    # ==============================================================================
    # Your data loading and t-SNE code should be above this line
    # ...
    # ==============================================================================
    # print("--- Quantitative Analysis of Feature Space Alignment (3 Domains) ---")

    # --- 1. Setup ---
    le = LabelEncoder()
    domain_labels_numeric = le.fit_transform(domain_labels)  # Will be 0, 1, 2

    # --- 2. Metric 1: Domain Classifier Accuracy ---
    # print("\nCalculating Domain Classifier Accuracy...")
    # An SVM can handle multi-class classification out of the box
    svm = SVC(kernel='linear', random_state=42)

    # Test on HIGH-DIMENSIONAL Raw Features
    acc_raw = cross_val_score(svm, flattend_all_patches, domain_labels_numeric, cv=5, scoring='accuracy')
    # print(f"Accuracy on Raw Features: {np.mean(acc_raw) * 100:.2f}% ± {np.std(acc_raw) * 100:.2f}%")

    # Test on HIGH-DIMENSIONAL LISA's Latent Features
    acc_ae = cross_val_score(svm, flattend_all_patchesAE, domain_labels_numeric, cv=5, scoring='accuracy')
    # print(f"Accuracy on LISA Latent Features: {np.mean(acc_ae) * 100:.2f}% ± {np.std(acc_ae) * 100:.2f}%")
    # print("(Closer to 33.3% is better, indicating domains are indistinguishable)")

    # --- 3. Metric 2: Average Pairwise MMD ---
    # print("\nCalculating Average Pairwise MMD...")

    # First, correctly slice the data into three distinct sets for pairwise comparison
    # Based on your domain_labels order: Lab, Field-AM, Field-PM
    lab_end_idx = num_lab_samples
    am_end_idx = num_lab_samples + num_am_samples

    # Slicing for Raw Features
    lab_features_raw = flattend_all_patches[:lab_end_idx]
    am_features_raw = flattend_all_patches[lab_end_idx:am_end_idx]
    pm_features_raw = flattend_all_patches[am_end_idx:]

    # Slicing for LISA's Latent Features
    lab_features_ae = flattend_all_patchesAE[:lab_end_idx]
    am_features_ae = flattend_all_patchesAE[lab_end_idx:am_end_idx]
    pm_features_ae = flattend_all_patchesAE[am_end_idx:]

    # --- MMD on HIGH-DIMENSIONAL Raw Features ---
    all_raw_distances = pairwise_distances(flattend_all_patches, metric='euclidean')
    median_dist_raw = np.median(all_raw_distances[all_raw_distances > 0])
    gamma_raw = 1.0 / (2 * median_dist_raw ** 2)

    mmd_raw_lab_am = mmd_rbf(lab_features_raw, am_features_raw, gamma=gamma_raw)
    mmd_raw_lab_pm = mmd_rbf(lab_features_raw, pm_features_raw, gamma=gamma_raw)
    mmd_raw_am_pm = mmd_rbf(am_features_raw, pm_features_raw, gamma=gamma_raw)
    avg_mmd_raw = (mmd_raw_lab_am + mmd_raw_lab_pm + mmd_raw_am_pm) / 3

    # print(f"Average Pairwise MMD on Raw Features (robust gamma): {avg_mmd_raw.item():.6f}")

    # --- MMD on HIGH-DIMENSIONAL LISA's Latent Features ---
    all_ae_distances = pairwise_distances(flattend_all_patchesAE, metric='euclidean')
    median_dist_ae = np.median(all_ae_distances[all_ae_distances > 0])
    gamma_ae = 1.0 / (2 * median_dist_ae ** 2)

    mmd_ae_lab_am = mmd_rbf(lab_features_ae, am_features_ae, gamma=gamma_ae)
    mmd_ae_lab_pm = mmd_rbf(lab_features_ae, pm_features_ae, gamma=gamma_ae)
    mmd_ae_am_pm = mmd_rbf(am_features_ae, pm_features_ae, gamma=gamma_ae)
    avg_mmd_ae = (mmd_ae_lab_am + mmd_ae_lab_pm + mmd_ae_am_pm) / 3

    # print(f"Average Pairwise MMD on LISA Latent Features (robust gamma): {avg_mmd_ae.item():.6f}")
    # print("(Lower is better)")

    # --- 4. Metric 3: Silhouette Score ---
    # print("\nCalculating Silhouette Score...")
    # The Silhouette Score also handles multiple labels automatically
    silhouette_raw = silhouette_score(flattend_all_patches, domain_labels_numeric)
    # print(f"Silhouette Score on Raw Features: {silhouette_raw:.4f}")

    silhouette_ae = silhouette_score(flattend_all_patchesAE, domain_labels_numeric)
    # print(f"Silhouette Score on LISA Latent Features: {silhouette_ae:.4f}")
    # print("(Closer to 0 or negative is better)")
    return (np.mean(acc_raw), np.mean(acc_ae), avg_mmd_raw.item(), avg_mmd_ae.item(), silhouette_raw, silhouette_ae)

# def get_latents(labo_dataset, scan_dataset, am_dataset, AE, Indexs):
#     labo_patches = []
#     scan_patches = []
#     am_patches = []
#     brix_values = []
#     all_brix_values = []
#     all_patches = []
#     labo_patches_AE = []
#     scan_patches_AE = []
#     am_patches_AE = []
#     all_patches_AE = []
#
#     for index in Indexs:
#         labo_patches.append(labo_dataset[index]['image'])
#         scan_patches.append(scan_dataset[index]['image'])
#         am_patches.append(am_dataset[index]['image'])
#         labo_patches_AE.append(AE(labo_dataset[index]['image'].unsqueeze(0).float())[0].detach())
#         scan_patches_AE.append(AE(scan_dataset[index]['image'].unsqueeze(0).float())[0].detach())
#         am_patches_AE.append(AE(am_dataset[index]['image'].unsqueeze(0).float())[0].detach())
#         brix_values.append(labo_dataset[index]['brix'].item())
#
#     all_patches.extend(labo_patches)
#     all_patches.extend(scan_patches)
#     all_patches.extend(am_patches)
#     all_patches_AE.extend(labo_patches_AE)
#     all_patches_AE.extend(scan_patches_AE)
#     all_patches_AE.extend(am_patches_AE)
#     all_patches_AE = np.array(all_patches_AE)
#     all_patches = np.array(all_patches)
#     labo_patches = np.array(labo_patches)
#     scan_patches = np.array(scan_patches)
#     all_brix_values.extend(brix_values.copy())
#     all_brix_values.extend(brix_values.copy())
#     all_brix_values.extend(brix_values.copy())
#
#     all_brix_values = np.array(all_brix_values)
#     all_brix_values = scan_dataset.scaler_labels_brix.inverse_transform(all_brix_values.reshape(-1, 1)).squeeze(1)
#
#     N = all_patches.shape[0]
#     flattend_all_patches = all_patches.reshape(N, -1)
#     flattend_all_patchesAE = all_patches_AE.reshape(N, -1)
#     # Using UMAP (recommended)
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30)
#     embedding_tsne = tsne.fit_transform(flattend_all_patches)
#     embedding_umap = reducer.fit_transform(flattend_all_patches)
#
#     reducer_ae = umap.UMAP(n_components=2, random_state=42)
#     tsne_ae = TSNE(n_components=2, random_state=42, perplexity=30)
#     embedding_AE_tsne = tsne_ae.fit_transform(flattend_all_patchesAE)
#     embedding_AE_umap = reducer_ae.fit_transform(flattend_all_patchesAE)
#     # Create domain labels for coloring
#     num_lab_samples = len(labo_patches)
#     num_scan_samples = len(scan_patches)
#     num_am_samples = len(am_patches)
#     domain_labels = ['Lab'] * num_lab_samples + ['Field-AM'] * num_am_samples + ['Field-PM'] * num_scan_samples
#     return domain_labels, flattend_all_patches, flattend_all_patchesAE, num_lab_samples, num_am_samples, embedding_tsne, embedding_umap, embedding_AE_tsne, embedding_AE_umap


def get_highest_metric_from_saved_models(models_dir, metric_name='testRecall'):
    """
    Scans a directory for saved models and finds the highest metric score
    based on a specific naming convention.

    Args:
        models_dir (str): The path to the directory containing saved models.
        metric_name (str): The name of the metric in the filename (e.g., 'testRecall').

    Returns:
        tuple: A tuple containing (highest_score, path_to_best_model).
               Returns (0.0, None) if no models are found.
    """
    highest_score = 0.0
    path_to_best_model = None

    # Regex to find the metric and its value in the directory name
    # e.g., 'yoloModel_wandb_..._testRecall_0.8512' -> extracts 0.8512
    pattern = re.compile(rf".*_{metric_name}_(\d+\.\d+)")

    if not os.path.exists(models_dir):
        return highest_score, path_to_best_model

    for dirname in os.listdir(models_dir):
        match = pattern.match(dirname)
        if match:
            try:
                score = float(match.group(1))
                if score > highest_score:
                    highest_score = score
                    path_to_best_model = os.path.join(models_dir, dirname)
            except (ValueError, IndexError):
                # Ignore directories that match the pattern but have an invalid number
                continue

    return highest_score, path_to_best_model