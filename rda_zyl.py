import numpy as np
import matplotlib.pyplot as plt

def generate_complex_chirp(length, f0, f1, T):
    t = np.linspace(0, T, length)
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t**2 / T)
    return np.exp(1j * phase)

# Load raw data image and transpose to correct axes
raw_image = plt.imread('ZylData/raw_data.png')
if raw_image.ndim == 3:
    raw_image = np.mean(raw_image, axis=2)  # Convert to grayscale
raw_image = raw_image.astype(np.float32) / 255.0  # Normalize

# Transpose to ensure rows=azimuth, columns=range (crucial step)
raw_data = raw_image.T.astype(np.complex64)  # Now rows=azimuth, columns=range

# Range Compression (horizontal direction)
num_range = raw_data.shape[1]  # Number of range samples
T_range = 1e-6  # Chirp duration (adjust based on your system)
f0_range, f1_range = 0, 5e6  # Chirp frequencies (adjust)

# Generate range chirp and matched filter
range_chirp = generate_complex_chirp(num_range, f0_range, f1_range, T_range)
range_ref = np.conj(range_chirp[::-1])  # Time-reversed conjugate

# Apply range compression to each azimuth line (row)
range_compressed = np.zeros_like(raw_data)
for i in range(raw_data.shape[0]):  # Iterate over azimuth positions
    signal_fft = np.fft.fft(raw_data[i, :])
    ref_fft = np.fft.fft(range_ref)
    range_compressed[i, :] = np.fft.ifft(signal_fft * ref_fft)

# Azimuth Compression (vertical direction)
num_azimuth = range_compressed.shape[0]  # Number of azimuth samples
T_azimuth = 1e-6  # Chirp duration (adjust)
f0_azimuth, f1_azimuth = 0, 5e6  # Chirp frequencies (adjust)

# Generate azimuth chirp and matched filter
azimuth_chirp = generate_complex_chirp(num_azimuth, f0_azimuth, f1_azimuth, T_azimuth)
azimuth_ref = np.conj(azimuth_chirp[::-1])

# Apply azimuth compression to each range bin (column)
sar_image = np.zeros_like(range_compressed)
for j in range(range_compressed.shape[1]):  # Iterate over range bins
    signal_fft = np.fft.fft(range_compressed[:, j])
    ref_fft = np.fft.fft(azimuth_ref)
    sar_image[:, j] = np.fft.ifft(signal_fft * ref_fft)

# Plot results (transpose back for correct display orientation)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(np.abs(raw_data.T), cmap='gray', aspect='auto')  # Transpose back
plt.title('Raw SAR Data\n(Range=Horizontal, Azimuth=Vertical)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.abs(range_compressed.T), cmap='gray', aspect='auto')  # Transpose back
plt.title('Range Compressed Data\n(Compressed Horizontally)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(sar_image.T), cmap='gray', aspect='auto')  # Transpose back
plt.title('Azimuth Compressed SAR Image\n(Final Focused Image)')
plt.axis('off')

plt.tight_layout()
plt.show()