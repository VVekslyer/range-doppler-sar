import os.path
import h5py
import matplotlib.pyplot as plt
import numpy as np

# use TeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

###################################
# USER PARAMETERS
###################################
# Choose HDF5 file
# sar_scene_folder = "PointTargetData"
# scene_name = "2024-12-09T125446" # Unburied point target

# sar_scene_folder = "PointTargetData/buried_target"
# scene_name = "2025-03-10T121609" # Plastic container target

sar_scene_folder = "PointTargetData/buried_target"
scene_name = "2025-03-10T122108" # Styrofoam buried target

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
filter_start = 400  # Changed from 500 to 400
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

# After displaying, crop to only keep range bins 0 to 400
crop_range_bins = filter_start
print(f"Cropping to use only range bins 0-{crop_range_bins}")
filtered_data_cropped = filtered_data[:, :crop_range_bins]

# Visualize the cropped data
plt.figure(figsize=(10, 6))
plt.imshow(filtered_data_cropped, aspect='auto', cmap='jet',
           vmin=vmin_rc, vmax=vmax_rc)
plt.title(f"Cropped Range-Compressed Data (0-{crop_range_bins} range bins)")
plt.xlabel("Range Bins")
plt.ylabel("Azimuth Samples")
plt.colorbar(label="Amplitude (dB)")
plt.tight_layout()
plt.show()

# Rotate the data 90 degrees clockwise at this early stage
filtered_data_rotated = np.rot90(filtered_data_cropped, k=-1)
print(f"Original data shape: {filtered_data_cropped.shape}, Rotated data shape: {filtered_data_rotated.shape}")

# Visualize the rotated data
plt.figure(figsize=(10, 6))
plt.imshow(filtered_data_rotated, aspect='auto', cmap='jet',
           vmin=vmin_rc, vmax=vmax_rc)
plt.title(f"Rotated Range-Compressed Data (90° CW)")
plt.xlabel("Azimuth Samples")
plt.ylabel("Range Bins")
plt.colorbar(label="Amplitude (dB)")
plt.tight_layout()
plt.show()

# Use rotated data for further processing
filtered_data_cropped = filtered_data_rotated  # Replace cropped data with rotated data

# Define radar parameters
ifSampleCount = 6000.0
ifGain = 15
mainDiv = 69.0
sweepBandwidth = 20E9  # 20 GHz bandwidth
sweepCenterFreq = 80E9 # 80 GHz center frequency
sweepCount = 3000
sweepDuration = 0.006    # 6 ms sweep
radarVelocity = 0.01     # 1 cm/s

# Calculate chirp rate (Hz/s) from sweep parameters
Kr = sweepBandwidth / sweepDuration  # Chirp rate

# Create range axis and frequency axis for the full range
τ_min = 0
τ_max = sweepDuration
τ_axis = np.linspace(τ_min, τ_max, num_range_samples)
dt = τ_axis[1] - τ_axis[0]
f_τ = np.fft.fftfreq(num_range_samples, d=dt)

# Create matched filter for range compression
G = np.exp(1j * np.pi * (f_τ**2) / Kr)

# After rotation, our dimensions have changed - need to adapt our approach
# Since we rotated, we now have (crop_range_bins, n_azimuth) instead of (n_azimuth, crop_range_bins)
print(f"Rotated data shape for processing: {filtered_data_cropped.shape}")

# Create matched filter appropriate for the rotated dimensions
# Now we need to match the second dimension (azimuth) instead of the first
new_range_bins = filtered_data_cropped.shape[1]  # This is now azimuth dimension after rotation
G_cropped = G[:new_range_bins]

# Convert filtered_data_cropped back to the complex domain for processing
filtered_spectrum = 10**(filtered_data_cropped/20) * np.exp(1j * 0)  # Convert from dB back to linear, using zero phase

# Apply the matched filter across the correct dimension (now axis=1)
# Reshape G_cropped to allow broadcasting with the rotated data
G_cropped_reshaped = G_cropped.reshape(1, -1)  # Shape (1, new_range_bins)
filtered_spectrum_matched = filtered_spectrum * G_cropped_reshaped  # Now shapes are compatible

# Reconstruct a symmetrically padded spectrum for the IFFT
full_spectrum = np.zeros((filtered_spectrum.shape[0], new_range_bins * 2), dtype=complex)
full_spectrum[:, :new_range_bins] = filtered_spectrum_matched
full_spectrum[:, new_range_bins:] = np.flip(np.conj(filtered_spectrum_matched), axis=1)

# Perform IFFT to get range-compressed signal - now operating on the rotated dimensions
s_rc = np.fft.ifft(full_spectrum, axis=1)

# ----- Step 4: Add window in range direction -----
print("\n----- Step 4: Window in Range Direction -----")
# Create window function adjusted for the rotated dimensions
range_window = np.hamming(new_range_bins * 2)

# Apply window across the appropriate dimension (now axis=1)
s_rc_windowed = s_rc * range_window.reshape(1, -1)

# Visualize the windowing effect
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Before windowing
im1 = axs[0].imshow(np.abs(s_rc), aspect='auto', cmap='jet')
axs[0].set_title("Range-Compressed Data (Before Windowing)")
axs[0].set_xlabel("Range Samples")
axs[0].set_ylabel("Azimuth Samples")
fig.colorbar(im1, ax=axs[0])

# After windowing
im2 = axs[1].imshow(np.abs(s_rc_windowed), aspect='auto', cmap='jet')
axs[1].set_title("Range-Compressed Data (After Windowing)")
axs[1].set_xlabel("Range Samples")
axs[1].set_ylabel("Azimuth Samples")
fig.colorbar(im2, ax=axs[1])

plt.tight_layout()
plt.show()

# ----- Step 5: FFT in range direction on windowed data -----
print("\n----- Step 5: FFT in Range Direction -----")
# Perform FFT in range direction on windowed data
S2 = np.fft.fft(s_rc_windowed, axis=1)

# Get the positive half of the spectrum to avoid the mirroring effect
half_spectrum_size = S2.shape[1] // 2
S2_positive = S2[:, :half_spectrum_size]

# Visualize both the full spectrum and the positive half
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Full spectrum (showing the mirroring issue)
S2_db_full = 20 * np.log10(np.abs(S2) + 1e-10)
im1 = axs[0].imshow(S2_db_full, aspect='auto', cmap='jet')
axs[0].set_title("Range Spectrum After Windowing (Full Spectrum)")
axs[0].set_xlabel("Frequency Bins")
axs[0].set_ylabel("Azimuth Samples")
fig.colorbar(im1, ax=axs[0], label="Magnitude (dB)")

# Positive half only
S2_db = 20 * np.log10(np.abs(S2_positive) + 1e-10)
im2 = axs[1].imshow(S2_db, aspect='auto', cmap='jet')
axs[1].set_title("Range Spectrum After Windowing (Positive Half Only)")
axs[1].set_xlabel("Frequency Bins")
axs[1].set_ylabel("Azimuth Samples")
fig.colorbar(im2, ax=axs[1], label="Magnitude (dB)")

plt.tight_layout()
plt.show()

# Use only the positive half for further processing
S2 = S2_positive

# ----- Step 6: FFT in azimuth direction + reference function in azimuth -----
print("\n----- Step 6: Azimuth FFT and Reference Function -----")
n_azimuth = S2.shape[0]

# Perform FFT in azimuth direction (with fftshift for visualization)
S3 = np.fft.fftshift(np.fft.fft(S2, axis=0), axes=0)

# Create azimuth axis and frequency axis
dt_az = sweepDuration / sweepCount  # Time between azimuth samples
f_η = np.fft.fftshift(np.fft.fftfreq(n_azimuth, d=dt_az))  # Azimuth frequency axis

# Reference function parameters - adjusted for point target focusing
c = 3e8  # Speed of light
λ = c / sweepCenterFreq  # Wavelength

# Improved parameter estimation for a point target scenario
# For a point reflector, we need to better estimate R0 and velocity effects
R0 = 5  # Adjust reference range based on expected target distance
v = 0.05  # Adjust velocity to 5 cm/s for better focus

# Create range-dependent azimuth reference functions
K_a = 2 * v**2 / (λ * R0)  # Azimuth FM rate

# Apply range-dependent azimuth focusing
# Create a 2D grid of reference functions to account for range dependence
range_bins = S3.shape[1]
H_a = np.zeros((n_azimuth, range_bins), dtype=complex)

# For each range bin, calculate appropriate reference function
for r in range(range_bins):
    # Scale R0 based on range bin (linear approximation)
    R_r = R0 * (1 + 0.1 * (r / range_bins))  # Adjust scale factor as needed
    K_a_r = 2 * v**2 / (λ * R_r)
    H_a[:, r] = np.exp(-1j * np.pi * (f_η**2) / K_a_r)

# Apply azimuth reference function to the 2D spectrum
S4 = S3 * H_a

# Visualize the data after azimuth reference function application
plt.figure(figsize=(10, 8))
S4_db = 20 * np.log10(np.abs(S4) + 1e-10)

S4_binary = np.zeros_like(S4_db)
S4_binary[S4_db > 127] = 1  # Changed from 130dB to 120dB
# Display with inverted binary colormap (white for high values, black for low values)
plt.imshow(S4_binary, aspect='auto', cmap='binary_r')  # binary_r is the reversed binary colormap
plt.title("2D Spectrum After Azimuth Reference Function (Binary)")
plt.xlabel("Range Frequency Bins")
plt.ylabel("Azimuth Frequency (Hz)")
plt.colorbar(label="Binary (Above/Below 127 dB)")

# Also show the original visualization with a jet colormap for comparison
plt.figure(figsize=(10, 8))
plt.imshow(S4_db, aspect='auto', cmap='jet')
plt.title("2D Spectrum After Azimuth Reference Function (Original)")
plt.xlabel("Range Frequency Bins")
plt.ylabel("Azimuth Frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")

plt.tight_layout()
plt.show()

# ----- Step 7: IFFT in azimuth direction -----
print("\n----- Step 7: Azimuth IFFT -----")
# Perform IFFT in azimuth direction
S5 = np.fft.ifft(np.fft.ifftshift(S4, axes=0), axis=0)

# Visualize the data after azimuth IFFT
plt.figure(figsize=(10, 8))
S5_db = 20 * np.log10(np.abs(S5) + 1e-10)
plt.imshow(S5_db, aspect='auto', cmap='jet')
plt.title("Data After Azimuth IFFT")
plt.xlabel("Range Frequency Bins")
plt.ylabel("Azimuth Samples")
plt.colorbar(label="Magnitude (dB)")
plt.tight_layout()
plt.show()

# ----- Step 8: Imaging (Final IFFT in range direction) -----
print("\n----- Step 8: Final Imaging -----")
# Perform IFFT in range direction for final image formation
final_image = np.fft.ifft(S5, axis=1)

# Extract magnitude for display
final_image_mag = np.abs(final_image)

# Apply additional post-processing to enhance point targets
# Apply log compression to enhance dynamic range
final_image_enhanced = np.log1p(final_image_mag)

# Crop the image to focus on the region of interest, but ensure we have valid dimensions
# This helps eliminate edge artifacts
n_azimuth, n_range_bins = final_image_enhanced.shape
print(f"Final image shape before cropping: {final_image_enhanced.shape}")

# Use more conservative cropping to ensure we don't end up with empty arrays
crop_range = min(int(n_range_bins * 0.5), n_range_bins - 2)  # Use 50% of range samples or ensure at least 2 remain
crop_azimuth = min(int(n_azimuth * 0.5), n_azimuth - 2)      # Use 50% of azimuth samples or ensure at least 2 remain
start_range = min(int(n_range_bins * 0.1), n_range_bins - crop_range - 1)  # Ensure we stay in bounds
start_azimuth = min(int(n_azimuth * 0.1), n_azimuth - crop_azimuth - 1)    # Ensure we stay in bounds

# Ensure we have valid crop dimensions
if crop_range <= 0 or crop_azimuth <= 0:
    print("Warning: Invalid crop dimensions. Using full image instead.")
    cropped_image = final_image_enhanced
    start_range, start_azimuth = 0, 0
    crop_range, crop_azimuth = n_range_bins, n_azimuth
else:
    cropped_image = final_image_enhanced[start_azimuth:start_azimuth+crop_azimuth, 
                                      start_range:start_range+crop_range]

print(f"Cropped image shape: {cropped_image.shape}")

# Create proper axes for the cropped image
# Adjust range resolution calculation - the original calculation might be too small
range_resolution = c / (2 * sweepBandwidth)  # Range resolution in meters
print(f"Original calculated range resolution: {range_resolution:.8f} meters")

# Calculate azimuth resolution
azimuth_resolution = λ / (2 * v * sweepDuration)  # Azimuth resolution in meters
print(f"Original calculated azimuth resolution: {azimuth_resolution:.8f} meters")

# Apply a more moderate scaling factor to avoid distortion
range_scaling = 100  # Reduced from 1000x to 100x for better visualization
range_extent = [0, crop_range * range_resolution * range_scaling]
azimuth_extent = [0, crop_azimuth * azimuth_resolution]

print(f"Range extent after scaling: {range_extent[0]:.2f} to {range_extent[1]:.2f} meters")
print(f"Azimuth extent: {azimuth_extent[0]:.2f} to {azimuth_extent[1]:.2f} meters")

# Display only the original SAR image without rotation
# plt.figure(figsize=(12, 10))
# plt.imshow(cropped_image, cmap='jet', aspect='equal',  # Changed to 'equal' for better proportions
#           extent=[range_extent[0], range_extent[1], azimuth_extent[1], azimuth_extent[0]])
# plt.title("SAR Image")
# plt.xlabel(f"Range (m) [scaled by {range_scaling}x]")
# plt.ylabel("Azimuth (m)")
# plt.colorbar(label="Log Magnitude")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()

# Create a cropped version of the SAR image focusing on the region from x=0 to x=200 and y=0 to y=200
plt.figure(figsize=(6, 5))

# Calculate the crop indices in the data based on the physical dimensions
max_x_value = range_extent[1]
max_y_value = azimuth_extent[1]

# Calculate what fraction of the total extent our desired crop represents
x_fraction = min(200 / max_x_value, 1.0)  # Limit to 1.0 if 200 exceeds the max value
y_fraction = min(200 / max_y_value, 1.0)  # Limit to 1.0 if 200 exceeds the max value

# Calculate corresponding indices in the image array
crop_x_pixels = int(cropped_image.shape[1] * x_fraction)
crop_y_pixels = int(cropped_image.shape[0] * y_fraction)

# Ensure we don't exceed array bounds
crop_x_pixels = min(crop_x_pixels, cropped_image.shape[1])
crop_y_pixels = min(crop_y_pixels, cropped_image.shape[0])

# Display the selected region
zoomed_image = cropped_image[:crop_y_pixels, :crop_x_pixels]
zoomed_extent = [0, 200, 0, 200]  # Fixed to 0-200 range for both x and y

plt.imshow(zoomed_image, cmap='jet', aspect='equal', 
          extent=zoomed_extent)
plt.title("SAR Image")
plt.xlabel(f"Range (m) [scaled by {range_scaling}x]")
plt.ylabel("Azimuth (m)")
plt.colorbar(label="Log Magnitude")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
