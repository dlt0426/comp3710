import numpy as np
import matplotlib.pyplot as plt

# Set parameters for the signal
N = 2048  # Number of sample points
T = 1.0   # Duration of the signal in seconds
f0 = 1.0  # Fundamental frequency of the square wave in Hz

# List of harmonic numbers used to construct the square wave
# harmonics = [1, 30, 50]
harmonics = [1, 3, 5]


def square_wave(t):
    return np.sign(np.sin(2.0 * np.pi * f0 * t))


def square_wave_fourier(t, f0, N):
    result = np.zeros_like(t)
    for k in range(N):
        n = 2 * k + 1
        result += np.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * result


# Create the time vector
# np.linspace generates evenly spaced numbers over a specified interval.
# We use endpoint=False because the interval is periodic.
t = np.linspace(0.0, T, N, endpoint=False)

# Generate the original square wave
square = square_wave(t)

plt.figure(figsize=(12, 8))

# Plot the original square wave
plt.subplot(2, 3, 1)
plt.plot(t, square, 'k', label='Square wave')
plt.title('Original Square Wave')
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.legend()

# Plot Fourier reconstructions under different numbers of harmonics
for i, Nh in enumerate(harmonics, start=2):
    plt.subplot(2, 3, i)
    y = square_wave_fourier(t, f0, Nh)
    plt.plot(t, y, label=f'N={Nh} harmonics')
    plt.plot(t, square, 'k--', alpha=0.5, label='Square wave')
    plt.title(f'Fourier Approximation with N={Nh}')
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# 2. Apply the DFT and time the execution
import time


def naive_dft(x):
    """Compute the discrete Fourier transform directly in O(N^2) time."""
    size = len(x)
    result = np.zeros(size, dtype=complex)
    for k in range(size):
        for n in range(size):
            angle = -2j * np.pi * k * n / size
            result[k] += x[n] * np.exp(angle)
    return result


signal = square_wave_fourier(t, f0, 50)

start_time_naive = time.time()
dft_result = naive_dft(signal)
naive_duration = time.time() - start_time_naive

start_time_fft = time.time()
fft_result = np.fft.fft(signal)
fft_duration = time.time() - start_time_fft

print("\n--- DFT / FFT Performance Comparison ---")
print(f"Naive DFT Execution Time: {naive_duration:.6f} seconds")
print(f"NumPy FFT Execution Time: {fft_duration:.6f} seconds")
if fft_duration > 0:
    print(f"FFT is approximately {naive_duration / fft_duration:.2f} times faster.")
else:
    print("FFT was too fast to measure a significant difference.")
print(f"\nOur DFT implementation is close to NumPy's FFT: {np.allclose(dft_result, fft_result)}")

# 4. Prepare for plotting
xf = np.fft.fftfreq(N, d=T / N)[:N // 2]
magnitude = 2.0 / N * np.abs(dft_result[:N // 2])

# 5. Visualize the results
plt.style.use("seaborn-v0_8-darkgrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
ax1.plot(t, signal, color="c")
ax1.set_title("Input Square Wave Signal", fontsize=16)
ax1.set_xlabel("Time (s)", fontsize=12)
ax1.set_ylabel("Amplitude", fontsize=12)
ax1.set_xlim(0, T)
ax1.grid(True)

ax2.stem(xf, magnitude, basefmt=" ")
ax2.set_title("Discrete Fourier Transform (Magnitude Spectrum)", fontsize=16)
ax2.set_xlabel("Frequency (Hz)", fontsize=12)
ax2.set_ylabel("Magnitude", fontsize=12)
ax2.set_xlim(0, 50)
ax2.grid(True)

for i in range(20):
    if i < len(xf) and i % 2 == 1:
        ax2.axvline(
            xf[i], color="r", linestyle="--", alpha=0.7,
            label=f"{i}*f0 = {xf[i]:.1f} Hz",
        )

ax2.legend()
plt.tight_layout()
plt.show()