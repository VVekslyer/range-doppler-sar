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
axs[0].set_title("2. Range Compressed Data (Real Part)")
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
# For a PRF of 1000 Hz like in the C&W book, dt_az is:
dt_az = 1/1000  # 0.5e-3 seconds (adjust if needed)

# Optionally, apply an azimuth window to reduce sidelobes.
azimuth_window = np.hanning(n_az)
s_rc_win = s_rc * azimuth_window[:, np.newaxis]

# Compute the azimuth FFT (along axis 0) with fftshift.
S1 = np.fft.fftshift(np.fft.fft(s_rc_win, axis=0), axes=0)

# Construct the azimuth frequency axis (in Hz)
f_η = np.fft.fftshift(np.fft.fftfreq(n_az, d=dt_az))

# For plotting, define an extent that maps:
# x-axis: expanded range axis (0 to 100 units)
# y-axis: Doppler frequency from f_eta.min() to f_eta.max()
extent_az = [new_x_min, new_x_max, f_η.min(), f_η.max()]
print("extent_az =", extent_az)

# Compute magnitude in dB for better dynamic range visualization.
mag_dB = 20 * np.log10(np.abs(S1) + 1e-12)

# Plot the azimuth FFT output: (a) Real part and (b) Magnitude (in dB)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Variation A: Real part of azimuth FFT
im0 = axs[0].imshow(np.real(S1), cmap='gray', extent=extent_az,
                    origin='lower', aspect='auto')
axs[0].set_title("3. Azimuth FFT (Real Part)")
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

# --------------------------------------------------------------------------------
# 4. Range Cell Migration Correction (RCMC)
# --------------------------------------------------------------------------------
# S1 is our azimuth FFT output (Doppler domain along azimuth, range dimension still in time).
# For RCMC, we need to work in the range frequency domain.

# Step 1: Transform the range (x-axis) to frequency domain.
S1_range = np.fft.fft(S1, axis=1)  # Now data is in (azimuth Doppler, range frequency)
print("S1_range =", S1_range.shape)

# Recompute (or re-use) the range frequency axis.
num_range_samples = s_rc.shape[1]  # number of range samples
f_tau = np.fft.fftfreq(num_range_samples, d=dt)  # 1D array, length = num_range_samples

# System parameters:
c = 3e8            # Speed of light (m/s)
f_c = 80e9         # Center frequency (80 GHz)
λ = c / f_c        # Wavelength (m), ~3.75e-3 m (3.75 mm)
# v and R0 should be defined from earlier processing; but I'll just use
v = 1              # Platform velocity in m/s
R0 = 10            # Reference slant range in meters

# Step 2: Compute the RCM amount for each azimuth frequency.
# f_eta (already computed) is the Doppler frequency axis (1D, length = n_az).
ΔR = (λ**2 * R0 * f_η**2) / (8 * v**2)  # shape: (n_az,)

# Step 3: Form the RCMC phase multiplier.
# For each azimuth frequency (row), multiplier is:
#   exp{ j*(4π/c)*f_tau*ΔR(f_eta) }.
# Use an outer product to generate a 2D matrix.
G_rcmc = np.exp(1j * 4 * np.pi / c * np.outer(ΔR, f_tau))  # shape: (n_az, num_range_samples)

# Step 4: Apply the RCMC correction in the range frequency domain.
S2 = S1_range * G_rcmc

# Step 5: Transform the corrected data back to the range time domain.
s2 = np.fft.ifft(S2, axis=1)

# Display the RCMC-corrected data: (a) Real part and (b) Magnitude, with brighter scaling and reversed colormap.
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Extract the real part and magnitude.
real_img = np.real(s2)
mag_img = np.abs(s2)

# Define color limits for brightening.
real_vmin = real_img.min()
real_vmax = real_img.mean() + 2*real_img.std()
mag_vmin = mag_img.min()
mag_vmax = mag_img.mean() + 2*mag_img.std()

# Plot the real part with reversed colormap.
im0 = axs[0].imshow(real_img, cmap='gray_r', aspect='auto',
                    vmin=real_vmin, vmax=real_vmax)
axs[0].set_title("4. RCMC Corrected Data (Real Part, Reversed)")
axs[0].set_xlabel("Expanded Range Axis (units)")
axs[0].set_ylabel("Azimuth (samples)")
fig.colorbar(im0, ax=axs[0])

# Plot the magnitude with reversed colormap.
im1 = axs[1].imshow(mag_img, cmap='gray_r', aspect='auto',
                    vmin=mag_vmin, vmax=mag_vmax)
axs[1].set_title("RCMC Corrected Data (Magnitude, Reversed)")
axs[1].set_xlabel("Expanded Range Axis (units)")
axs[1].set_ylabel("Azimuth (samples)")
fig.colorbar(im1, ax=axs[1])

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# 5. Azimuth Compression & Azimuth IFFT
# --------------------------------------------------------------------------------

# Compute the azimuth FM rate, K_a (approximately 2V_r^2/(λR0))
K_a = 2 * v**2 / (λ * R0)  

# --- Azimuth FFT ---
# Compute the number of azimuth samples and the azimuth sampling interval.
n_az = s2.shape[0]

# Perform the azimuth FFT of the RCMC-corrected data (s2) along axis=0.
S_az = np.fft.fftshift(np.fft.fft(s2, axis=0), axes=0)

# Construct the Doppler frequency axis (in Hz).
f_η = np.fft.fftshift(np.fft.fftfreq(n_az, d=dt_az))

# --- Azimuth Matched Filtering ---
# The azimuth matched filter is given by:
#    H_az(f_eta) = exp{ -j * π * (f_eta^2) / K_a }
H_az = np.exp(-1j * np.pi * (f_η**2) / K_a)

# Apply the matched filter (broadcast H_az along the range dimension).
S3 = S_az * H_az[:, None]

# Inverse FFT along azimuth (axis=0) to obtain the final focused image.
S_ac = np.fft.ifft(np.fft.ifftshift(S3, axes=0), axis=0)
final_image = np.abs(S_ac)

# --- For the spectrum plot, compute the range frequency axis ---
num_range_samples = s2.shape[1]
f_tau_shift = np.fft.fftshift(np.fft.fftfreq(num_range_samples, d=dt))

# --- Define extents for plotting ---
# For the final focused image, we map the range axis to an "expanded" coordinate.
new_x_min = 0
new_x_max = 100  # Expanded range axis (units)
extent_final = [new_x_min, new_x_max, n_az, 0]  # x: expanded range, y: azimuth sample index

# For the Doppler spectrum, set extent using f_tau and f_eta.
extent_spec = [f_tau_shift.min(), f_tau_shift.max(), f_η.min(), f_η.max()]

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# (a) Compressed signal: Final focused image (amplitude) with Azimuth (samples) vs. Range time (samples).
im_a = axs[0, 0].imshow(final_image, cmap='gray_r', extent=extent_final, aspect='auto')
axs[0, 0].set_title("5. Final Focused Image (Compressed Signal)")
axs[0, 0].set_xlabel("Expanded Range Axis (units)")
axs[0, 0].set_ylabel("Azimuth (samples)")
fig.colorbar(im_a, ax=axs[0, 0])

# (b) Target C Spectrum: Magnitude (in dB) of S3 (Doppler-domain data) with Azimuth freq vs. Range freq.
im_b = axs[0, 1].imshow(20*np.log10(np.abs(S3)+1e-12), cmap='gray_r', extent=extent_spec,
                         origin='lower', aspect='auto')
axs[0, 1].set_title("Target C Spectrum (S₃) in dB")
axs[0, 1].set_xlabel("Range Frequency (Hz)")
axs[0, 1].set_ylabel("Azimuth Frequency (Hz)")
fig.colorbar(im_b, ax=axs[0, 1])

# (c) Expanded Target C: Final focused image with expanded range axis.
im_c = axs[1, 0].imshow(final_image, cmap='gray_r', extent=extent_final, aspect='auto')
axs[1, 0].set_title("Expanded Final Focused Image")
axs[1, 0].set_xlabel("Expanded Range Axis (units)")
axs[1, 0].set_ylabel("Azimuth (samples)")
fig.colorbar(im_c, ax=axs[1, 0])

# (d) Expanded Target C Contours: Contour plot of the final focused image.
azimuth_idx = np.arange(n_az)
range_expanded = np.linspace(new_x_min, new_x_max, num_range_samples)
X, Y = np.meshgrid(range_expanded, azimuth_idx)
cs = axs[1, 1].contour(X, Y, final_image, levels=10, cmap='gray_r')
axs[1, 1].set_title("Expanded Final Focused Image (Contours)")
axs[1, 1].set_xlabel("Expanded Range Axis (units)")
axs[1, 1].set_ylabel("Azimuth (samples)")
fig.colorbar(cs.collections[0], ax=axs[1, 1])

plt.tight_layout()
plt.show()
