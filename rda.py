import os.path
import h5py
import matplotlib.pyplot as plt
import numpy as np

###################################
# USER PARAMETERS
###################################
# Update the path to point to your HDF5 file
sar_scene_folder = "PointTargetData"
scene_name = "2024-12-09T125446"

print(f"Looking for files in: {os.path.abspath(sar_scene_folder)}")
print("Files in directory:")
for file in os.listdir(sar_scene_folder):
    print(f"- {file}")


data_name = os.path.join(sar_scene_folder, scene_name + ".hdf5")
print(f"Using file: {data_name}")

# ----- Step 1: Raw Data -----
# Open the HDF5 file
with h5py.File(data_name, "r") as data:
    #radar_frequencies = ["80_GHz_Radar", "144_GHz_Radar", "240_GHz_Radar"]
    radar_frequencies = ["80_GHz_Radar"] 
    
    for radar in radar_frequencies:
        raw_data = data[f"{radar}/raw_adc"][()]
        
        # Calculate dB values for raw data display
        raw_db = 20 * np.log10(np.abs(raw_data) + 1e-10)
        
        # Create a single figure for raw data visualization
        plt.figure(figsize=(10, 8))
        
        # Raw ADC data visualization
        vmin_raw, vmax_raw = np.percentile(raw_db, [5, 95])
        plt.imshow(raw_db, aspect='auto', cmap='viridis', vmin=vmin_raw, vmax=vmax_raw)
        plt.title(f"{radar} - Raw ADC Data")
        plt.xlabel("Range Samples")
        plt.ylabel("Azimuth Samples")
        plt.colorbar(label="Amplitude (dB)")
        
        plt.tight_layout()
        plt.show()

# Process raw data for Step 2 (Range Compression) - use absolute values since we may have real data encoded oddly
raw_data_abs = np.abs(raw_data)

# ----- Step 2: Range Compression -----
print("\n----- Step 2: Range Compression -----")
num_range_samples = raw_data_abs.shape[1]

# Perform range FFT - The main range compression processing
S0 = np.fft.fft(raw_data_abs, axis=1)

# Show range-compressed data before and after filtering
# Show only half of the range compressed data to avoid mirroring
n_range = S0.shape[1]
half_range = n_range // 2
half_range_compressed = S0[:, :half_range]
range_compressed_db = 20 * np.log10(np.abs(half_range_compressed) + 1e-10)

# Create filtered version
filtered_data = range_compressed_db.copy()
threshold_db = 65 # Threshold value
filter_start = 500
filtered_data[:, filter_start:] = np.minimum(filtered_data[:, filter_start:], threshold_db)

# Create a figure with two vertically stacked plots for range compression results
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot 1: Range-compressed data before filtering
vmin_rc, vmax_rc = np.percentile(range_compressed_db, [10, 99.5])
im1 = axs[0].imshow(range_compressed_db, aspect='auto', cmap='jet', 
                  vmin=vmin_rc, vmax=vmax_rc)
axs[0].set_title(f"{radar} - Range-Compressed Data (Before Filtering)")
axs[0].set_xlabel("Range Bins")
axs[0].set_ylabel("Azimuth Samples")
fig.colorbar(im1, ax=axs[0], label="Amplitude (dB)")

# Plot 2: Range-compressed data after filtering
im2 = axs[1].imshow(filtered_data, aspect='auto', cmap='jet', 
                  vmin=vmin_rc, vmax=vmax_rc)  # Same color scale as Plot 1
axs[1].set_title(f"Range-Compressed Data (After Filtering, >{threshold_db}dB capped beyond bin {filter_start})")
axs[1].set_xlabel("Range Bins")
axs[1].set_ylabel("Azimuth Samples")
fig.colorbar(im2, ax=axs[1], label="Amplitude (dB)")

plt.tight_layout()
plt.show()

# Apply ideal low-pass filter and continue with further processing
# ...existing code...

# Process raw data for Step 2 (Real Range Compression) - use absolute values since we may have real data encoded oddly
raw_data_abs = np.abs(raw_data)

# ----- Step 2: Range Compression (Main Processing) -----
print("\n----- Step 2: Range Compression -----")
num_range_samples = raw_data_abs.shape[1]

# Perform range FFT - This is the actual range compression for processing
S0 = np.fft.fft(raw_data_abs, axis=1)

# Apply ideal low-pass filter (as shown in the example)
cutoff = int(0.2 * num_range_samples)  # Tunable parameter
print(f"Using frequency cutoff at {cutoff}/{num_range_samples} samples")

# Create a filtered version (keeping only lower frequencies)
S0_filtered = S0.copy()
S0_filtered[:, cutoff:] = 0

# Alternative: Apply matched filter instead of simple low-pass filter
τ_min = -2e-6    # seconds
τ_max = 1.5e-5   # seconds
τ_axis = np.linspace(τ_min, τ_max, num_range_samples)
dt = τ_axis[1] - τ_axis[0]
f_τ = np.fft.fftfreq(num_range_samples, d=dt)

Kr = 1e12  # Hz/s: assumed chirp rate
G = np.exp(1j * np.pi * (f_τ**2) / Kr)

# Apply matched filter
S0_matched = S0 * G

# Perform IFFT for both methods to compare
s_rc_lowpass = np.fft.ifft(S0_filtered, axis=1)
s_rc_matched = np.fft.ifft(S0_matched, axis=1)

# For plotting: expand the range axis from 0 to 100 units.
new_x_min = 0
new_x_max = 100
y_min = 0
y_max = s_rc_lowpass.shape[0]

# Compare both methods - show the full range compressed data
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Show the range compressed data after applying the different processing methods
im0 = axs[0].imshow(np.abs(s_rc_lowpass), cmap='gray')
axs[0].set_title("Low-pass Filtered Range Compression")
axs[0].set_xlabel("Range Samples")
axs[0].set_ylabel("Azimuth (samples)")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(np.abs(s_rc_matched), cmap='gray')
axs[1].set_title("Matched Filter Range Compression")
axs[1].set_xlabel("Range Samples")
axs[1].set_ylabel("Azimuth (samples)")
plt.colorbar(im1, ax=axs[1])

plt.tight_layout()
plt.show()

# Choose which range-compressed data to use for subsequent steps
# s_rc = s_rc_lowpass  # Uncomment to use low-pass filtering
s_rc = s_rc_matched   # Uncomment to use matched filtering

# ----- Step 3: Azimuth FFT -----
n_az = s_rc.shape[0]
dt_az = 1/1000  # Azimuth sampling interval in seconds; adjust if needed
azimuth_window = np.hanning(n_az)
s_rc_win = s_rc * azimuth_window[:, np.newaxis]
S1 = np.fft.fftshift(np.fft.fft(s_rc_win, axis=0), axes=0)
f_eta = np.fft.fftshift(np.fft.fftfreq(n_az, d=dt_az))
extent_az = [new_x_min, new_x_max, f_eta.min(), f_eta.max()]
mag_dB = 20 * np.log10(np.abs(S1) + 1e-12)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
im0 = axs[0].imshow(np.real(S1), cmap='gray', extent=extent_az, origin='lower', aspect='auto')
axs[0].set_title("3. Azimuth FFT (Real Part)")
axs[0].set_xlabel("Expanded Range Axis (units)")
axs[0].set_ylabel("Azimuth Frequency (Hz)")
plt.colorbar(im0, ax=axs[0])
im1 = axs[1].imshow(mag_dB, cmap='gray', extent=extent_az, origin='lower', aspect='auto')
axs[1].set_title("Azimuth FFT (Magnitude in dB)")
axs[1].set_xlabel("Expanded Range Axis (units)")
axs[1].set_ylabel("Azimuth Frequency (Hz)")
plt.colorbar(im1, ax=axs[1])
plt.tight_layout()
plt.show()

# ----- Step 4: Range Cell Migration Correction (RCMC) -----
S1_range = np.fft.fft(S1, axis=1)
num_range_samples = s_rc.shape[1]
f_tau = np.fft.fftfreq(num_range_samples, d=dt)

# Radar/system parameters (adjust as needed)
c = 3e8
f_c = 80e9
λ = c / f_c
v = 1    # Platform velocity (m/s)
R0 = 10  # Reference slant range (m)

ΔR = (λ**2 * R0 * f_eta**2) / (8 * v**2)
G_rcmc = np.exp(1j * 4 * np.pi / c * np.outer(ΔR, f_tau))
S2 = S1_range * G_rcmc
s2 = np.fft.ifft(S2, axis=1)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
real_img = np.real(s2)
mag_img = np.abs(s2)
im0 = axs[0].imshow(real_img, cmap='gray_r', aspect='auto')
axs[0].set_title("4. RCMC Corrected Data (Real Part, Reversed)")
axs[0].set_xlabel("Expanded Range Axis (units)")
axs[0].set_ylabel("Azimuth (samples)")
plt.colorbar(im0, ax=axs[0])
im1 = axs[1].imshow(mag_img, cmap='gray_r', aspect='auto')
axs[1].set_title("RCMC Corrected Data (Magnitude, Reversed)")
axs[1].set_xlabel("Expanded Range Axis (units)")
axs[1].set_ylabel("Azimuth (samples)")
plt.colorbar(im1, ax=axs[1])
plt.tight_layout()
plt.show()

# ----- Step 5: Azimuth Compression & IFFT -----
K_a = 2 * v**2 / (λ * R0)
S_az = np.fft.fftshift(np.fft.fft(s2, axis=0), axes=0)
f_eta = np.fft.fftshift(np.fft.fftfreq(n_az, d=dt_az))
H_az = np.exp(-1j * np.pi * (f_eta**2) / K_a)
S3 = S_az * H_az[:, None]
S_ac = np.fft.ifft(np.fft.ifftshift(S3, axes=0), axis=0)
final_image = np.abs(S_ac)

num_range_samples = s2.shape[1]
f_tau_shift = np.fft.fftshift(np.fft.fftfreq(num_range_samples, d=dt))
extent_final = [new_x_min, new_x_max, n_az, 0]
extent_spec = [f_tau_shift.min(), f_tau_shift.max(), f_eta.min(), f_eta.max()]

fig, axs = plt.subplots(2, 2, figsize=(14, 12))
im_a = axs[0, 0].imshow(final_image, cmap='gray_r', extent=extent_final, aspect='auto')
axs[0, 0].set_title("5. Final Focused Image (Compressed Signal)")
axs[0, 0].set_xlabel("Expanded Range Axis (units)")
axs[0, 0].set_ylabel("Azimuth (samples)")
plt.colorbar(im_a, ax=axs[0, 0])
im_b = axs[0, 1].imshow(20*np.log10(np.abs(S3)+1e-12), cmap='gray_r', extent=extent_spec, origin='lower', aspect='auto')
axs[0, 1].set_title("Target C Spectrum (S₃) in dB")
axs[0, 1].set_xlabel("Range Frequency (Hz)")
axs[0, 1].set_ylabel("Azimuth Frequency (Hz)")
plt.colorbar(im_b, ax=axs[0, 1])
im_c = axs[1, 0].imshow(final_image, cmap='gray_r', extent=extent_final, aspect='auto')
axs[1, 0].set_title("Expanded Final Focused Image")
axs[1, 0].set_xlabel("Expanded Range Axis (units)")
axs[1, 0].set_ylabel("Azimuth (samples)")
plt.colorbar(im_c, ax=axs[1, 0])
azimuth_idx = np.arange(n_az)
range_expanded = np.linspace(new_x_min, new_x_max, num_range_samples)
X, Y = np.meshgrid(range_expanded, azimuth_idx)
cs = axs[1, 1].contour(X, Y, final_image, levels=10, cmap='gray_r')
axs[1, 1].set_title("Expanded Final Focused Image (Contours)")
axs[1, 1].set_xlabel("Expanded Range Axis (units)")
axs[1, 1].set_ylabel("Azimuth (samples)")
plt.colorbar(cs.collections[0], ax=axs[1, 1])
plt.tight_layout()
plt.show()