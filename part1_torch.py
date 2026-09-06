"""
COMP3710 Lab 2 - Part 1: DFT with NumPy vs PyTorch (CPU/GPU)

This script:
  1. Rewrites square_wave, square_wave_fourier and naive_dft using PyTorch.
  2. Adds a GPU version of naive_dft implemented as a matrix multiply
     (X = W @ x), NOT using torch.fft.
  3. Benchmarks three methods across several signal sizes:
       - NumPy naive DFT      (CPU, O(N^2), python double loop)
       - NumPy FFT            (CPU, O(N log N), optimised)
       - PyTorch naive DFT    (GPU, O(N^2), parallel matrix multiply)
  4. Saves the harmonic-reconstruction and spectrum plots to PNG files.

Run on Rangpur with sbatch (A100). Plots are saved, not shown, because
compute nodes have no display.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend: save figures instead of showing them
import matplotlib.pyplot as plt
import torch

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

T = 1.0    # signal duration (seconds)
f0 = 1.0   # fundamental frequency (Hz)


# ----------------------------------------------------------------------
# 1. NumPy versions (kept for the CPU baseline / comparison)
# ----------------------------------------------------------------------
def square_wave_np(t):
    return np.sign(np.sin(2.0 * np.pi * f0 * t))


def square_wave_fourier_np(t, f0, n_harmonics):
    result = np.zeros_like(t)
    for k in range(n_harmonics):
        n = 2 * k + 1                       # odd harmonics only
        result += np.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * result


def naive_dft_np(x):
    """Direct O(N^2) DFT in NumPy (python double loop)."""
    size = len(x)
    result = np.zeros(size, dtype=complex)
    for k in range(size):
        for n in range(size):
            angle = -2j * np.pi * k * n / size
            result[k] += x[n] * np.exp(angle)
    return result


# ----------------------------------------------------------------------
# 2. PyTorch versions
# ----------------------------------------------------------------------
def square_wave_torch(t):
    return torch.sign(torch.sin(2.0 * np.pi * f0 * t))


def square_wave_fourier_torch(t, f0, n_harmonics):
    result = torch.zeros_like(t)
    for k in range(n_harmonics):
        n = 2 * k + 1
        result += torch.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * result


def naive_dft_torch(x):
    """
    Naive O(N^2) DFT as a matrix multiply: X = W @ x, where
    W[k, n] = exp(-2j*pi*k*n/N).

    This runs on whatever device x lives on. On the GPU the whole N x N
    matrix multiply is parallelised across the CUDA cores, instead of
    being a serial python loop. We deliberately do NOT use torch.fft.
    """
    size = x.shape[0]
    dev = x.device
    k = torch.arange(size, device=dev).reshape(size, 1)   # column vector
    n = torch.arange(size, device=dev).reshape(1, size)   # row vector
    angle = -2.0 * np.pi * k * n / size                   # N x N real angles
    W = torch.exp(1j * angle)                             # N x N complex matrix
    x_c = x.to(torch.complex128)
    return W.to(torch.complex128) @ x_c


# ----------------------------------------------------------------------
# 3. Harmonic reconstruction plot (accuracy & sharpness)
# ----------------------------------------------------------------------
N_plot = 2048
t_np = np.linspace(0.0, T, N_plot, endpoint=False)
square = square_wave_np(t_np)
harmonics = [1, 3, 5, 20, 50]   # includes the higher-order cases the lab asks for

plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.plot(t_np, square, "k", label="Square wave")
plt.title("Original Square Wave")
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.legend()

for i, Nh in enumerate(harmonics, start=2):
    plt.subplot(2, 3, i)
    y = square_wave_fourier_np(t_np, f0, Nh)
    plt.plot(t_np, y, label=f"N={Nh} harmonics")
    plt.plot(t_np, square, "k--", alpha=0.5, label="Square wave")
    plt.title(f"Fourier Approximation N={Nh}")
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig("harmonics.png", dpi=120)
plt.close()
print("Saved harmonics.png")


# ----------------------------------------------------------------------
# 4. Correctness check: torch GPU DFT vs numpy FFT on a small signal
# ----------------------------------------------------------------------
N_check = 512
t_check = np.linspace(0.0, T, N_check, endpoint=False)
sig_np = square_wave_fourier_np(t_check, f0, 50)
sig_torch = torch.from_numpy(sig_np).to(device)

gpu_dft = naive_dft_torch(sig_torch).cpu().numpy()
fft_ref = np.fft.fft(sig_np)
print("torch GPU DFT close to numpy FFT:", np.allclose(gpu_dft, fft_ref, atol=1e-3))


# ----------------------------------------------------------------------
# 5. Three-way timing benchmark across data sizes
# ----------------------------------------------------------------------
def time_numpy_naive(sig_np):
    start = time.time()
    naive_dft_np(sig_np)
    return time.time() - start


def time_numpy_fft(sig_np):
    start = time.time()
    np.fft.fft(sig_np)
    return time.time() - start


def time_torch_gpu(sig_torch):
    # warm-up (first CUDA call includes one-off setup cost)
    naive_dft_torch(sig_torch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    naive_dft_torch(sig_torch)
    if device.type == "cuda":
        torch.cuda.synchronize()   # wait for GPU to actually finish before stopping timer
    return time.time() - start


sizes = [256, 512, 1024, 2048]
print("\n--- Timing across data sizes (seconds) ---")
print(f"{'N':>6} | {'NumPy naive':>12} | {'NumPy FFT':>12} | {'Torch GPU DFT':>14}")
print("-" * 55)

for N in sizes:
    tt = np.linspace(0.0, T, N, endpoint=False)
    s_np = square_wave_fourier_np(tt, f0, 50)
    s_torch = torch.from_numpy(s_np).to(device)

    d_naive = time_numpy_naive(s_np)
    d_fft = time_numpy_fft(s_np)
    d_gpu = time_torch_gpu(s_torch)

    print(f"{N:>6} | {d_naive:>12.6f} | {d_fft:>12.6f} | {d_gpu:>14.6f}")

    # order fastest -> slowest for this N
    methods = {"NumPy naive": d_naive, "NumPy FFT": d_fft, "Torch GPU DFT": d_gpu}
    order = sorted(methods.items(), key=lambda kv: kv[1])
    print("        fastest -> slowest:",
          " < ".join(f"{name} ({t:.4f}s)" for name, t in order))

print("\nDone.")
