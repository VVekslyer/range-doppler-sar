# Trying to use the data and step-by-step process of Cummings & Wong Chapter 6
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
# 1. Raw Data: Load and plot the original SAR raw data images
# --------------------------------------------------------------------------------
# Load the images
real_part = plt.imread('SimulationData/real_part_raw.png')
imag_part = plt.imread('SimulationData/imag_part_raw.png')
phase = plt.imread('SimulationData/phase_raw.png')

# Convert images to grayscale if needed (drop alpha channel if present)
def rgb2gray(img):
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    else:
        return img

real_gray = rgb2gray(real_part)
imag_gray = rgb2gray(imag_part)

# Normalize values to [0, 1]
real_norm = (real_gray - np.min(real_gray)) / (np.max(real_gray) - np.min(real_gray))
imag_norm = (imag_gray - np.min(imag_gray)) / (np.max(imag_gray) - np.min(imag_gray))

# Reconstruct the complex SAR raw data s0(τ,η)
s0 = real_norm + 1j * imag_norm

# Plot the raw data as a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
im0 = axs[0, 0].imshow(np.real(s0), cmap='gray')
axs[0, 0].set_title("Real Part")
axs[0, 0].set_xlabel("Range (samples)")
axs[0, 0].set_ylabel("Azimuth (samples)")

im1 = axs[0, 1].imshow(np.imag(s0), cmap='gray')
axs[0, 1].set_title("Imaginary Part")
axs[0, 1].set_xlabel("Range (samples)")
axs[0, 1].set_ylabel("Azimuth (samples)")

im2 = axs[1, 0].imshow(phase, cmap='gray')
axs[1, 0].set_title("Phase")
axs[1, 0].set_xlabel("Range (samples)")
axs[1, 0].set_ylabel("Azimuth (samples)")

im3 = axs[1, 1].imshow(np.abs(s0), cmap='gray')
axs[1, 1].set_title("Magnitude")
axs[1, 1].set_xlabel("Range (samples)")
axs[1, 1].set_ylabel("Azimuth (samples)")

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# 2. Range Compression: FFT along range, matched filtering, and IFFT back to time.
# --------------------------------------------------------------------------------
num_range_samples = s0.shape[1]
τ_min = -2e-6    # seconds
τ_max = 1.5e-5   # seconds
τ_axis = np.linspace(τ_min, τ_max, num_range_samples)
dt = τ_axis[1] - τ_axis[0]

# Compute the frequency axis for the range dimension
f_τ = np.fft.fftfreq(num_range_samples, d=dt)

# Define the chirp rate used in your simulation
Kr = 1e12  # Hz/s

# Define the frequency-domain matched filter
G = np.exp(1j * np.pi * (f_τ**2) / Kr)

# Perform FFT along the range dimension (axis=1)
S0 = np.fft.fft(s0, axis=1)
# Multiply by the matched filter in frequency domain
S0_filtered = S0 * G
# Inverse FFT to obtain range-compressed data s_rc(τ,η)
s_rc = np.fft.ifft(S0_filtered, axis=1)

# For plotting, we want the x-axis (range) coordinate to span 0 to 100 units.
new_x_min = 0
new_x_max = 100  # expanded coordinate span for range
y_min = 0
y_max = s_rc.shape[0]

# Plot range-compressed data (Real and Magnitude)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

im0 = axs[0].imshow(np.real(s_rc), cmap='gray',
                    extent=[new_x_min, new_x_max, y_max, y_min])
axs[0].set_title("Range Compressed Data (Real Part)")
axs[0].set_xlabel("Expanded Range Axis (units)")
axs[0].set_ylabel("Azimuth (samples)")
axs[0].set_aspect('auto')  # Changed from 'equal' to 'auto'
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(np.abs(s_rc), cmap='gray',
                    extent=[new_x_min, new_x_max, y_max, y_min])
axs[1].set_title("Range Compressed Data (Magnitude)")
axs[1].set_xlabel("Expanded Range Axis (units)")
axs[1].set_ylabel("Azimuth (samples)")
axs[1].set_aspect('auto')  # Changed from 'equal' to 'auto'
fig.colorbar(im1, ax=axs[1])

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# 3. Azimuth FFT: Transform each range gate (column) into the Doppler domain.
# --------------------------------------------------------------------------------
# Determine the number of azimuth samples from the range-compressed data.
n_az = s_rc.shape[0]

# Set the azimuth sampling interval based on the radar PRF.
# For a PRF of 2000 Hz, dt_az is:
dt_az = 1/2000  # 0.5e-3 seconds (adjust if needed)

# Optionally, apply an azimuth window to reduce sidelobes.
azimuth_window = np.hanning(n_az)
s_rc_win = s_rc * azimuth_window[:, np.newaxis]

# Compute the azimuth FFT (along axis 0) with fftshift.
S1 = np.fft.fftshift(np.fft.fft(s_rc_win, axis=0), axes=0)

# Construct the azimuth frequency axis (in Hz)
f_eta = np.fft.fftshift(np.fft.fftfreq(n_az, d=dt_az))

# For plotting, define an extent that maps:
# x-axis: expanded range axis (0 to 100 units)
# y-axis: Doppler frequency from f_eta.min() to f_eta.max()
extent_az = [new_x_min, new_x_max, f_eta.min(), f_eta.max()]

# Compute magnitude in dB for better dynamic range visualization.
mag_dB = 20 * np.log10(np.abs(S1) + 1e-12)

# Plot the azimuth FFT output: (a) Real part and (b) Magnitude (in dB)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Variation A: Real part of azimuth FFT
im0 = axs[0].imshow(np.real(S1), cmap='gray', extent=extent_az,
                    origin='lower', aspect='auto')
axs[0].set_title("Azimuth FFT (Real Part)")
axs[0].set_xlabel("Expanded Range Axis (units)")
axs[0].set_ylabel("Azimuth Frequency (Hz)")
fig.colorbar(im0, ax=axs[0])

# Variation B: Magnitude (in dB) of azimuth FFT
im1 = axs[1].imshow(mag_dB, cmap='gray', extent=extent_az,
                    origin='lower', aspect='auto')
axs[1].set_title("Azimuth FFT (Magnitude in dB)")
axs[1].set_xlabel("Expanded Range Axis (units)")
axs[1].set_ylabel("Azimuth Frequency (Hz)")
fig.colorbar(im1, ax=axs[1])

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# Additional Azimuth Matched Filtering Using System Parameters
# --------------------------------------------------------------------------------
# System parameters:
# c = 3e8
# fc = 80e9           # Center frequency: 80 GHz
# lam = c / fc        # Wavelength (m)
# v = 0.0166          # Platform velocity in m/s
# R0 = 2.0            # Assumed reference slant range in meters

# # Compute the azimuth FM rate (Ka); for a broadside system:
# Ka = - (2 * v**2) / (lam * R0)  
# # (Ka is negative if Doppler frequency decreases with time)

# # Generate the azimuth time vector (eta), centered at zero.
# eta = np.linspace(-n_az/2, n_az/2 - 1, n_az) * dt_az  # in seconds

# # The expected azimuth phase in the raw data (from target motion) is:
# # exp(-j*pi*|Ka|*eta^2).  Therefore, the matched filter (reference chirp)
# # should be the complex conjugate: exp(+j*pi*|Ka|*eta^2).
# ref_chirp = np.exp(1j * np.pi * np.abs(Ka) * eta**2)

# # Transform the reference chirp to the Doppler domain.
# H_ref = np.fft.fftshift(np.fft.fft(ref_chirp))

# # Apply the matched filter in the Doppler domain to each range bin.
# # (H_ref is 1D; multiply along axis 0.)
# S1_focused = S1 * np.conj(H_ref)[:, np.newaxis]

# # Optionally, perform an inverse FFT to obtain the focused azimuth time domain signal.
# focused_azimuth = np.fft.ifft(np.fft.ifftshift(S1_focused, axes=0), axis=0)

# # For visualization, compute the magnitude (in dB) of the focused Doppler data.
# mag_focused_dB = 20 * np.log10(np.abs(S1_focused) + 1e-12)

# # Plot the focused azimuth FFT result.
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# im0 = axs[0].imshow(np.real(focused_azimuth), cmap='gray', extent=extent_az,
#                     origin='lower', aspect='auto')
# axs[0].set_title("Focused Azimuth (Real Part)")
# axs[0].set_xlabel("Expanded Range Axis (units)")
# axs[0].set_ylabel("Azimuth Frequency (Hz)")
# fig.colorbar(im0, ax=axs[0])

# im1 = axs[1].imshow(mag_focused_dB, cmap='gray', extent=extent_az,
#                     origin='lower', aspect='auto')
# axs[1].set_title("Focused Azimuth (Magnitude in dB)")
# axs[1].set_xlabel("Expanded Range Axis (units)")
# axs[1].set_ylabel("Azimuth Frequency (Hz)")
# fig.colorbar(im1, ax=axs[1])

# plt.tight_layout()
# plt.show()

