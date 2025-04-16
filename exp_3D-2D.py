import numpy as np
import h5py
import matplotlib.pyplot as plt


# === Load .hdf5 2D SAR Data ===
filename = r"C:\Users\Owner\Documents\GitHub\range-doppler-sar\PointTargetData\2024-12-09T125446.hdf5"  # Change this to your actual path

with h5py.File(filename, 'r') as f:
    raw_adc = f['80_GHz_Radar/raw_adc'][()]  # Shape: [slow-time, fast-time] or vice versa

# Transpose if needed: want shape = [fast-time (range), slow-time (along-track)]
if raw_adc.shape[0] < raw_adc.shape[1]:
    raw_adc = raw_adc.T

print("Loaded raw_adc shape:", raw_adc.shape)
num_range_bins, num_positions = raw_adc.shape

# === Optional: Background Subtraction ===
# If you have a background file or background frame(s), subtract it here
# raw_adc = raw_adc - raw_background

# === Windowing (Optional) ===
raw_adc = raw_adc * np.hamming(num_range_bins)[:, None]

# === Range FFT (fast-time) ===
range_profiles = np.fft.fft(raw_adc, axis=0)

# === Azimuth FFT (slow-time) ===
sar_image = np.fft.fftshift(np.fft.fft(range_profiles, axis=1), axes=1)

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.imshow(20 * np.log10(np.abs(sar_image) / np.max(np.abs(sar_image))),
           cmap='gray', aspect='auto', extent=[0, num_positions, 0, num_range_bins])
plt.title("2D SAR Image (Range vs Along-Track)")
plt.xlabel("Along-Track Position (slow time index)")
plt.ylabel("Range Bin (fast time index)")
plt.colorbar(label='dB')
plt.tight_layout()
plt.show()