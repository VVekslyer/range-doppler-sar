import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D

def fmcw_sar_simulation():
    # System Parameters
    c = 3e8                  # Speed of light (m/s)
    fc = 5e9                 # Carrier frequency (5 GHz)
    B = 800e6                # Sweep bandwidth (800 MHz)
    Tp = 5e-3                # Sweep period (5 ms)
    alpha = B/Tp             # Chirp rate (Hz/s)
    lambda_c = c/fc          # Wavelength (m)
    
    # Geometry
    R_ref = 100              # Reference range to scene center (m)
    v = 1.0                  # Platform velocity (m/s)
    beamwidth = 10.5 * np.pi/180  # Azimuth beam width (rad)
    
    # Target positions (x, y) - where x is range, y is azimuth
    targets = np.array([[95, 0],   # Target 1: (range, azimuth)
                       [105, 5]])  # Target 2: (range, azimuth)
    
    # Synthetic aperture length
    L_a = 2 * R_ref * np.tan(beamwidth/2)  # Synthetic aperture length
    print(f"Synthetic aperture length: {L_a:.2f} m")
    
    # Discretization
    PRF = 500                # Pulse Repetition Frequency (Hz)
    num_pulses = 1024        # Number of pulses (azimuth samples)
    num_samples = 1024       # Number of samples per pulse (range samples)
    
    # Time arrays
    t_m = np.linspace(-L_a/(2*v), L_a/(2*v), num_pulses)  # Slow time (azimuth time)
    t_hat = np.linspace(0, Tp, num_samples)               # Fast time (range time)
    
    # Calculate range resolution and azimuth resolution
    range_res = c/(2*B)  # Range resolution
    azimuth_res = lambda_c/(2*beamwidth)  # Azimuth resolution
    print(f"Range resolution: {range_res:.3f} m")
    print(f"Azimuth resolution: {azimuth_res:.3f} m")
    
    # Generate raw data (after dechirping)
    print("\nStep 1: Generating dechirped signal + phase modified function")
    raw_data = np.zeros((num_pulses, num_samples), dtype=complex)
    
    for i, tm in enumerate(t_m):
        for j, tt in enumerate(t_hat):
            # For each target
            for target in targets:
                # Range to target
                R0 = target[0]  # Range at closest approach
                y0 = target[1]  # Azimuth position
                
                # Calculate instantaneous slant range
                R_t = np.sqrt(R0**2 + (v*tm - y0)**2)
                
                # Doppler frequency shift calculation
                f_d = -2/lambda_c * (v**2 * (tm - y0/v)) / R_t
                
                # Signal model from equation (10)
                # First exponential term - phase due to range
                phase1 = -4 * np.pi * R_t / lambda_c
                
                # Second exponential term - phase due to dechirping
                phase2 = -4 * np.pi * alpha * (R_t - R_ref) * (tt - 2*R_ref/c) / c
                
                # Phase modified function h1 from equation (11)
                phase_h1 = -4 * np.pi * lambda_c * f_d * tt / (2 * lambda_c) - 4 * np.pi * alpha * lambda_c * f_d * tt * (tt - 2*R_ref/c) / (2 * c)
                
                # Combined signal
                signal_phase = phase1 + phase2 + phase_h1
                raw_data[i, j] += np.exp(1j * signal_phase)
    
    # Display raw data
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(raw_data), aspect='auto', cmap='jet')
    plt.title("Raw Data (after dechirping + phase modified function)")
    plt.xlabel("Range Samples")
    plt.ylabel("Azimuth Samples")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()
    
    # Step 2: Range FFT + De-sloping factor
    print("\nStep 2: Range FFT + De-sloping factor")
    range_fft = np.fft.fft(raw_data, axis=1)
    
    # Display range compressed data
    plt.figure(figsize=(10, 8))
    range_compressed_db = 20 * np.log10(np.abs(range_fft) + 1e-10)
    plt.imshow(range_compressed_db, aspect='auto', cmap='jet')
    plt.title("Range Compressed Data (FFT in Range Direction)")
    plt.xlabel("Range Frequency Bins")
    plt.ylabel("Azimuth Samples")
    plt.colorbar(label="Amplitude (dB)")
    plt.tight_layout()
    plt.show()
    
    # Step 3: IFFT in Range Direction
    print("\nStep 3: IFFT in Range Direction")
    range_ifft = np.fft.ifft(range_fft, axis=1)
    
    # Display range IFFT data
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(range_ifft), aspect='auto', cmap='jet')
    plt.title("Data After Range IFFT")
    plt.xlabel("Range Samples")
    plt.ylabel("Azimuth Samples")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()
    
    # Step 4: Adding Window in Range Direction
    print("\nStep 4: Adding Window in Range Direction")
    # Create Hamming window
    range_window = np.hamming(num_samples)
    
    # Apply window
    windowed_data = range_ifft * range_window[np.newaxis, :]
    
    # Display windowed data
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(windowed_data), aspect='auto', cmap='jet')
    plt.title("Windowed Data in Range Direction")
    plt.xlabel("Range Samples")
    plt.ylabel("Azimuth Samples")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()
    
    # Step 5: FFT in Range Direction Again
    print("\nStep 5: FFT in Range Direction")
    range_fft2 = np.fft.fft(windowed_data, axis=1)
    
    # Display range FFT data
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(range_fft2), aspect='auto', cmap='jet')
    plt.title("Data After Second Range FFT")
    plt.xlabel("Range Frequency Bins")
    plt.ylabel("Azimuth Samples")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()
    
    # Step 6: FFT in Azimuth Direction + Reference Function
    print("\nStep 6: FFT in Azimuth Direction + Reference Function")
    # FFT in azimuth direction
    azimuth_fft = np.fft.fftshift(np.fft.fft(range_fft2, axis=0), axes=0)
    
    # Create azimuth reference function (Equation 17)
    freq_az = np.fft.fftshift(np.fft.fftfreq(num_pulses, d=t_m[1]-t_m[0]))
    H_az = np.zeros((num_pulses, num_samples), dtype=complex)
    
    for i, fa in enumerate(freq_az):
        for j in range(num_samples):
            # Reference function from equation (17)
            phase = -np.pi * fa**2 * R_ref * c / (2 * v**2 * (fc - 4*alpha*R_ref/c))
            H_az[i, j] = np.exp(1j * phase)
    
    # Apply azimuth reference function
    azimuth_compressed = azimuth_fft * H_az
    
    # Display azimuth compressed data
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(azimuth_compressed), aspect='auto', cmap='jet')
    plt.title("Azimuth Compressed Data (FFT + Reference Function)")
    plt.xlabel("Range Frequency Bins")
    plt.ylabel("Azimuth Frequency Bins")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()
    
    # Step 7: IFFT in Azimuth Direction
    print("\nStep 7: IFFT in Azimuth Direction")
    azimuth_ifft = np.fft.ifft(np.fft.ifftshift(azimuth_compressed, axes=0), axis=0)
    
    # Display azimuth IFFT data
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(azimuth_ifft), aspect='auto', cmap='jet')
    plt.title("Data After Azimuth IFFT")
    plt.xlabel("Range Frequency Bins")
    plt.ylabel("Azimuth Samples")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()
    
    # Step 8: Final Imaging
    print("\nStep 8: Final Imaging")
    # IFFT in range direction for the final image
    final_image = np.fft.ifft(azimuth_ifft, axis=1)
    final_image_mag = np.abs(final_image)
    
    # Dynamic range adjustment for better visualization
    final_image_db = 20 * np.log10(final_image_mag + 1e-10)
    vmin = np.percentile(final_image_db, 10)
    vmax = np.percentile(final_image_db, 99.5)
    
    # Create proper axis scales
    range_axis = np.linspace(R_ref - 20, R_ref + 20, num_samples)  # Range axis in meters
    azimuth_axis = np.linspace(-L_a/2, L_a/2, num_pulses)  # Azimuth axis in meters
    
    # Display final SAR image with proper scales
    plt.figure(figsize=(12, 10))
    plt.imshow(final_image_db, aspect='auto', cmap='jet',
               extent=[range_axis[0], range_axis[-1], azimuth_axis[-1], azimuth_axis[0]],
               vmin=vmin, vmax=vmax)
    plt.title("Final SAR Image")
    plt.xlabel("Range (m)")
    plt.ylabel("Azimuth (m)")
    plt.colorbar(label="Magnitude (dB)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot the actual target positions for reference
    plt.scatter(targets[:, 0], targets[:, 1], c='r', marker='x', s=100, label='True Target Positions')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Create 3D surface plot for final visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(range_axis, azimuth_axis)
    Z = final_image_mag
    
    surf = ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=False)
    ax.set_title("3D Surface Plot of SAR Image")
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Azimuth (m)')
    ax.set_zlabel('Intensity')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()
    
    return final_image_mag, range_axis, azimuth_axis, targets

if __name__ == "__main__":
    final_image, range_axis, azimuth_axis, targets = fmcw_sar_simulation()
    print("\nSimulation complete!")
