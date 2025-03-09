import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def range_doppler_algorithm(raw_data, radar_params):
    """
    Implements the Range-Doppler Algorithm (RDA)
    
    Steps included:
    1. Raw radar data
    2. Range Compression without the IFFT
    3. Azimuth FFT to Range-Doppler domain
    4. SRC and Range IFFT
    5. RCMC (Range Cell Migration Correction)
    
    Args:
        raw_data: 2D complex array of raw SAR data (azimuth × range)
        radar_params: Dictionary containing radar parameters
        
    Returns:
        range_doppler: Data after SRC and Range IFFT
    """
    print("Starting Range-Doppler Algorithm (Steps 1-5)...")
    n_azimuth, n_range = raw_data.shape
    print(f"Data dimensions: {n_azimuth} azimuth samples × {n_range} range samples")
    
    ##############################################
    # STEP 1-2: RANGE COMPRESSION (WITHOUT IFFT)
    ##############################################
    print("Step 1-2: Range Compression (FFT + MF)")
    # Apply window to reduce range sidelobes
    range_window = np.hamming(n_range)
    
    # FFT along range dimension
    range_fft = np.fft.fft(raw_data * range_window[np.newaxis, :], axis=1)
    
    # Create range matched filter
    range_mf = create_range_matched_filter(n_range, radar_params)
    
    # Apply range matched filter (frequency domain multiplication)
    range_compressed_freq = range_fft * range_mf[np.newaxis, :]
     
    # Note: We intentionally skip the IFFT step here
    
    ##############################################
    # STEP 3: TRANSFORM TO RANGE-DOPPLER DOMAIN (AZIMUTH FFT)
    ##############################################
    print("Step 3: Transforming to Range-Doppler Domain")
    # Apply window to reduce azimuth sidelobes
    azimuth_window = np.hamming(n_azimuth)
    
    # FFT along azimuth dimension and shift zero-frequency to center
    range_doppler_freq = np.fft.fftshift(
        np.fft.fft(range_compressed_freq * azimuth_window[:, np.newaxis], axis=0), 
        axes=0
    )
    
    ##############################################
    # STEP 4: SRC AND RANGE IFFT
    ##############################################
    print("Step 4: Secondary Range Compression and Range IFFT")

    # Extract parameters for SRC
    wavelength = radar_params['wavelength']    # m
    v = radar_params['platform_velocity']      # m/s
    r0 = radar_params['range_to_center']       # m
    fs = radar_params['sampling_rate']         # Hz
    bandwidth = radar_params['bandwidth']      # Hz 
    pulse_duration = radar_params['pulse_duration']  # s

    # Calculate nominal chirp rate
    kr_0 = bandwidth / pulse_duration

    # Doppler frequency axis (centered)
    doppler_freq = np.fft.fftshift(np.fft.fftfreq(n_azimuth, radar_params['pri']))

    # Range frequency axis (unshifted)
    range_freq = np.fft.fftfreq(n_range, 1/fs)

    # Diagnostic information
    print(f"Nominal chirp rate: {kr_0:.3e} Hz/s")
    print(f"Wavelength: {wavelength:.3e} m")
    print(f"Platform velocity: {v} m/s")
    print(f"Max Doppler frequency: {np.max(np.abs(doppler_freq)):.3f} Hz")
    print(f"Max (wavelength*fd/(2*v))^2 value: {np.max((wavelength * np.abs(doppler_freq) / (2 * v))**2):.6f}")

    # Initialize output array
    src_corrected = np.zeros_like(range_doppler_freq)

    # Safety threshold for numerical stability
    safety_threshold = 0.95

    # Apply SRC for each Doppler bin
    skipped_bins = 0
    for i, fd in enumerate(doppler_freq):
        # Calculate term inside sqrt - this represents the effect of platform motion
        inside_term = (wavelength * fd / (2 * v))**2
        
        # Safety check to avoid numerical instability
        if inside_term >= safety_threshold:
            # Skip this bin (use uncorrected data)
            src_corrected[i, :] = range_doppler_freq[i, :]
            skipped_bins += 1
            continue
        
        # Calculate effective chirp rate modification factor based on Doppler
        D_fd = 1 / np.sqrt(1 - (wavelength * fd / (2 * v))**2)

        # Calculate modified chirp rate for this Doppler bin
        Kr_fd = kr_0 * D_fd

        # Create phase correction based on chirp rate difference
        # This is directly related to the QPE formula mentioned
        T_r = pulse_duration  # Pulse duration (often Tr/2 is used in calculations)
        phase = np.pi * range_freq**2 * (1/kr_0 - 1/Kr_fd)

        # Apply phase correction
        src_corrected[i, :] = range_doppler_freq[i, :] * np.exp(-1j * phase)

    print(f"Skipped {skipped_bins} out of {n_azimuth} Doppler bins due to numerical concerns")

    # Now perform range IFFT to complete SRC
    range_doppler = np.fft.ifft(src_corrected, axis=1)

    ##############################################
    # STEP 5: RANGE CELL MIGRATION CORRECTION (RCMC)
    ##############################################
    print("Step 5: Range Cell Migration Correction")

    # Extract parameters needed for RCMC
    wavelength = radar_params['wavelength']    # m
    v = radar_params['platform_velocity']      # m/s
    r0 = radar_params['range_to_center']       # m
    fs = radar_params['sampling_rate']         # Hz
    c = 3e8                                   # Speed of light (m/s)
    range_spacing = c / (2 * fs)              # Range sample spacing (m)

    # Doppler frequency axis (centered)
    doppler_freq = np.fft.fftshift(np.fft.fftfreq(n_azimuth, radar_params['pri']))

    # Create output array for RCMC-corrected data
    rcmc_corrected = np.zeros_like(range_doppler)

    # Diagnostic info
    print(f"Range sample spacing: {range_spacing:.6f} m")
    max_rcm = (wavelength**2 * r0 * np.max(np.abs(doppler_freq))**2) / (8 * v**2)
    print(f"Maximum range migration: {max_rcm:.6f} m ({max_rcm/range_spacing:.2f} samples)")

    # Use scipy's interpolation instead of custom sinc interpolation
    from scipy import interpolate

    # Set maximum shift limit to avoid excessive interpolation
    MAX_SHIFT_SAMPLES = 200  # Limit the maximum shift to a reasonable value

    # Create range bins array once
    range_bins = np.arange(n_range)

    # Apply RCMC for each Doppler bin
    for i, fd in enumerate(doppler_freq):
        # Calculate range cell migration for this Doppler frequency
        # Formula: ΔR(fη) = (λ² R₀ fη²)/(8Vr²)
        range_migration = (wavelength**2 * r0 * fd**2) / (8 * v**2)
        
        # Convert migration from meters to samples
        sample_shift = range_migration / range_spacing
        
        if abs(sample_shift) < 0.01:
            # Skip small shifts
            rcmc_corrected[i, :] = range_doppler[i, :]
            continue
        
        # Limit the maximum shift to avoid excessive interpolation
        if abs(sample_shift) > MAX_SHIFT_SAMPLES:
            if i % 100 == 0:  # Avoid excessive printing
                print(f"Warning: Limiting shift at bin {i} from {sample_shift:.2f} to {MAX_SHIFT_SAMPLES if sample_shift > 0 else -MAX_SHIFT_SAMPLES} samples")
            sample_shift = MAX_SHIFT_SAMPLES if sample_shift > 0 else -MAX_SHIFT_SAMPLES
        
        # Shifted sample positions (negative shift because we're correcting the migration)
        shifted_positions = range_bins - sample_shift
        
        # Use scipy's interpolation for better performance
        # Handle real and imaginary parts separately
        real_part = np.real(range_doppler[i, :])
        imag_part = np.imag(range_doppler[i, :])
        
        # Create interpolation functions (cubic spline for better accuracy)
        real_interp = interpolate.interp1d(
            range_bins, real_part, kind='cubic', 
            bounds_error=False, fill_value=0
        )
        
        imag_interp = interpolate.interp1d(
            range_bins, imag_part, kind='cubic', 
            bounds_error=False, fill_value=0
        )
        
        # Apply interpolation
        valid_shifts = (shifted_positions >= 0) & (shifted_positions < n_range)
        rcmc_corrected[i, valid_shifts] = real_interp(shifted_positions[valid_shifts]) + 1j * imag_interp(shifted_positions[valid_shifts])
        
        if i % 100 == 0:  # Print progress occasionally
            print(f"Processed Doppler bin {i}/{n_azimuth}, shift = {sample_shift:.2f} samples")

    print(f"RCMC completed")

    # Return both SRC and RCMC results
    return {"src_result": range_doppler, "rcmc_result": rcmc_corrected}

def create_range_matched_filter(n_range, radar_params):
    """
    Create range matched filter for pulse compression.
    The matched filter is the complex conjugate of the expected signal (chirp).
    """
    # Extract parameters
    bandwidth = radar_params['bandwidth']      # Hz
    pulse_duration = radar_params['pulse_duration']  # s
    fs = radar_params['sampling_rate']         # Hz
    
    # Frequency axis
    freq = np.fft.fftfreq(n_range, 1/fs)
    
    # Chirp rate (Hz/s)
    chirp_rate = bandwidth / pulse_duration
    
    # Matched filter in frequency domain (complex conjugate of expected signal)
    # For a linear FM chirp, the phase is quadratic in frequency
    phase = -np.pi * freq**2 / chirp_rate
    
    # Create matched filter with frequency-domain phase
    range_mf = np.exp(1j * phase)
    
    # Apply spectral weighting to reduce sidelobes (e.g., Hamming window)
    window = np.hamming(n_range)
    range_mf *= window
    
    return range_mf

def apply_src(range_doppler_data, radar_params):
    """
    Apply Secondary Range Compression (SRC) in the Range-Doppler domain.
    
    SRC corrects for the variation of the FM rate with Doppler frequency.
    This is critical for wide-beam or high-resolution SAR systems.
    """
    n_azimuth, n_range = range_doppler_data.shape
    
    # Extract parameters
    wavelength = radar_params['wavelength']    # m
    v = radar_params['platform_velocity']      # m/s
    r0 = radar_params['range_to_center']       # m
    fs = radar_params['sampling_rate']         # Hz
    c = 3e8                                    # Speed of light (m/s)
    
    # Doppler frequency axis (centered)
    doppler_freq = np.fft.fftshift(np.fft.fftfreq(n_azimuth, radar_params['pri']))
    
    # Range time axis
    range_time = np.arange(n_range) / fs
    
    # Initialize output array
    src_corrected = np.zeros_like(range_doppler_data)
    
    # Apply SRC for each Doppler bin
    for i, fd in enumerate(doppler_freq):
        # Range curvature factor based on Doppler frequency
        # This accounts for how the effective chirp rate changes with Doppler
        curvature_factor = 1 / np.sqrt(1 - (wavelength * fd / (2 * v))**2)
        
        # Phase correction for SRC
        # This corrects the quadratic phase error in range that varies with Doppler
        phase = 2 * np.pi * (curvature_factor - 1) * range_time * fd**2 * wavelength**2 / (8 * v**2)
        
        # Apply phase correction to this Doppler bin
        src_corrected[i, :] = range_doppler_data[i, :] * np.exp(-1j * phase)
    
    return src_corrected

def apply_rcmc(range_doppler_data, radar_params):
    """
    Apply Range Cell Migration Correction (RCMC) in the Range-Doppler domain.
    
    RCMC corrects for the range migration that occurs because the distance 
    to targets changes during the synthetic aperture.
    """
    n_azimuth, n_range = range_doppler_data.shape
    
    # Extract parameters
    wavelength = radar_params['wavelength']    # m
    v = radar_params['platform_velocity']      # m/s
    r0 = radar_params['range_to_center']       # m
    fs = radar_params['sampling_rate']         # Hz
    c = 3e8                                    # Speed of light (m/s)
    
    # Doppler frequency axis (centered)
    doppler_freq = np.fft.fftshift(np.fft.fftfreq(n_azimuth, radar_params['pri']))
    
    # Range bin size (m)
    range_bin_size = c / (2 * fs)
    
    # Initialize output array
    rcmc_corrected = np.zeros_like(range_doppler_data, dtype=complex)
    
    # Create interpolator for each azimuth line
    range_bins = np.arange(n_range)
    
    # Apply RCMC for each Doppler bin
    for i, fd in enumerate(doppler_freq):
        # Calculate range migration in meters
        # This is the range curvature due to the synthetic aperture
        range_shift = (wavelength**2 * r0 * fd**2) / (8 * v**2)
        
        # Convert to bin shift
        bin_shift = range_shift / range_bin_size
        
        # Interpolation to correct range migration
        # Shift range bins by the calculated amount
        shifted_bins = range_bins - bin_shift
        
        # Use sinc interpolation for better accuracy (via cubic spline approximation)
        # Only interpolate where we have valid data
        valid_indices = (shifted_bins >= 0) & (shifted_bins <= n_range - 1)
        if np.any(valid_indices):
            real_interp = interpolate.interp1d(
                shifted_bins[valid_indices], 
                np.real(range_doppler_data[i, valid_indices]),
                kind='cubic', 
                bounds_error=False, 
                fill_value=0
            )
            
            imag_interp = interpolate.interp1d(
                shifted_bins[valid_indices], 
                np.imag(range_doppler_data[i, valid_indices]),
                kind='cubic', 
                bounds_error=False, 
                fill_value=0
            )
            
            rcmc_corrected[i, :] = real_interp(range_bins) + 1j * imag_interp(range_bins)
    
    return rcmc_corrected

def create_azimuth_matched_filter(data_shape, radar_params):
    """
    Create azimuth matched filter for azimuth compression in Range-Doppler domain.
    
    The filter phase is based on the Doppler frequency history of a point target.
    """
    n_azimuth, n_range = data_shape
    
    # Extract parameters
    wavelength = radar_params['wavelength']    # m
    v = radar_params['platform_velocity']      # m/s
    r0 = radar_params['range_to_center']       # m
    pri = radar_params['pri']                  # s
    
    # Doppler frequency axis (centered)
    doppler_freq = np.fft.fftshift(np.fft.fftfreq(n_azimuth, pri))
    
    # Initialize output array
    azimuth_mf = np.zeros((n_azimuth, n_range), dtype=complex)
    
    # Calculate range for each range bin
    range_axis = np.linspace(r0 - n_range/4, r0 + n_range/4, n_range)  # Approximate range values
    
    # Calculate azimuth matched filter for each range bin
    for j, r in enumerate(range_axis):
        # Doppler rate depends on range (rad/s²)
        # This is how fast the Doppler frequency changes with azimuth position
        doppler_rate = 2 * v**2 / (wavelength * r)
        
        # Azimuth matched filter phase
        # This is the quadratic phase needed to focus the azimuth signal
        phase = -np.pi * doppler_freq**2 / doppler_rate
        
        # Apply the matched filter for this range bin
        azimuth_mf[:, j] = np.exp(1j * phase)
    
    return azimuth_mf

def estimate_radar_parameters(radar_frequency):
    """
    Estimate radar parameters based on frequency.
    
    For a real system, these would come from system specifications or calibration.
    """
    # Convert radar frequency name to actual frequency
    freq_dict = {
        "80_GHz_Radar": 80e9,    # 80 GHz in Hz
        "144_GHz_Radar": 144e9,  # 144 GHz in Hz
        "240_GHz_Radar": 240e9   # 240 GHz in Hz
    }
    
    center_freq = freq_dict.get(radar_frequency, 100e9)  # Default to 100 GHz if not found
    wavelength = 3e8 / center_freq  # Calculate wavelength (m)
    
    # Estimate parameters - these are reasonable values for a lab-based SAR setup
    params = {
        'center_frequency': center_freq,    # Hz
        'wavelength': wavelength,           # m
        'bandwidth': 2e9,                   # 2 GHz bandwidth
        'pulse_duration': 1e-6,             # 1 μs pulse duration
        'sampling_rate': 6e9,               # 6 GHz sampling rate (based on 6000 range samples)
        'pri': 1e-3,                        # 1 ms pulse repetition interval (1 kHz PRF)
        'platform_velocity': 0.1,           # 0.1 m/s (typical for lab scanner)
        'range_to_center': 1.0,             # 1 meter (typical for lab setup)
    }
    
    return params

def plot_sar_processing_steps(raw_data, range_compressed, final_image, radar_frequency):
    """
    Plot the SAR processing steps: raw data, range compressed data, and final SAR image.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    
    # Plot 1: Raw data magnitude in dB
    raw_db = 20 * np.log10(np.abs(raw_data) + 1e-10)
    vmin_raw, vmax_raw = np.percentile(raw_db, [5, 98])
    im1 = axes[0].imshow(raw_db, aspect='auto', cmap='viridis', vmin=vmin_raw, vmax=vmax_raw)
    axes[0].set_title(f"{radar_frequency} - Raw SAR Data")
    axes[0].set_xlabel("Range Samples")
    axes[0].set_ylabel("Azimuth Samples")
    plt.colorbar(im1, ax=axes[0], label="Amplitude (dB)")
    
    # Plot 2: Range compressed data
    range_db = 20 * np.log10(np.abs(range_compressed) + 1e-10)
    vmin_rc, vmax_rc = np.percentile(range_db, [5, 98])
    im2 = axes[1].imshow(range_db, aspect='auto', cmap='jet', vmin=vmin_rc, vmax=vmax_rc)
    axes[1].set_title(f"{radar_frequency} - Range-Compressed Data")
    axes[1].set_xlabel("Range Bins")
    axes[1].set_ylabel("Azimuth Samples")
    plt.colorbar(im2, ax=axes[1], label="Amplitude (dB)")

    # Plot 3: Final SAR image
    final_db = 20 * np.log10(np.abs(final_image) + 1e-10)
    vmin_final, vmax_final = np.percentile(final_db, [5, 98])
    im3 = axes[2].imshow(final_db, aspect='auto', cmap='gray', vmin=vmin_final, vmax=vmax_final)
    axes[2].set_title(f"{radar_frequency} - Final SAR Image")
    axes[2].set_xlabel("Range Samples")
    axes[2].set_ylabel("Azimuth Samples")
    plt.colorbar(im3, ax=axes[2], label="Amplitude (dB)")

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to process SAR data using the Range-Doppler Algorithm.
    This function handles file loading, data extraction, and calls the RDA processing.
    """
    # User parameters - same as in plot_raw_data.py for consistency
    sar_scene_folder = "PointTargetData"
    scene_name = "2024-12-09T125446"
    
    print(f"Looking for SAR data in: {os.path.abspath(sar_scene_folder)}")
    
    # Try different file extensions
    extensions = [".h5", ".hdf5", ".hdf"]
    found_file = False
    
    for ext in extensions:
        data_path = os.path.join(sar_scene_folder, scene_name + ext)
        if os.path.exists(data_path):
            found_file = True
            print(f"Found SAR data file: {data_path}")
            
            try:
                with h5py.File(data_path, "r") as data:
                    print("File opened successfully. Available radar frequencies:")
                    radar_frequencies = ["80_GHz_Radar", "144_GHz_Radar", "240_GHz_Radar"]
                    
                    # Process each radar dataset
                    for radar in radar_frequencies:
                        if f"{radar}/raw_adc" in data:
                            print(f"\n{'='*50}")
                            print(f"Processing {radar} data")
                            print(f"{'='*50}")
                            
                            # Extract raw SAR data
                            raw_data = data[f"{radar}/raw_adc"][()]
                            
                            # If data is not complex, convert to complex format
                            # (Assuming data might be stored as real or integer values)
                            if not np.iscomplexobj(raw_data):
                                print("Converting data to complex format...")
                                raw_data = raw_data.astype(np.complex128)
                            
                            # Estimate radar parameters based on frequency
                            radar_params = estimate_radar_parameters(radar)
                            print(f"Estimated radar parameters:")
                            for key, value in radar_params.items():
                                print(f"  {key}: {value}")
                            
                            # Step 1: Range Compression (first part of RDA)
                            print("\nPerforming initial range compression...")
                            range_window = np.hamming(raw_data.shape[1])
                            range_fft = np.fft.fft(raw_data * range_window[np.newaxis, :], axis=1)
                            range_mf = create_range_matched_filter(raw_data.shape[1], radar_params)
                            range_compressed_freq = range_fft * range_mf[np.newaxis, :]
                            range_compressed = np.fft.ifft(range_compressed_freq, axis=1)
                            
                            # Perform partial Range-Doppler processing (up to Step 4)
                            print("Starting Range-Doppler Algorithm (Steps 1-5)...")
                            results = range_doppler_algorithm(raw_data, radar_params)
                            range_doppler = results["src_result"]
                            rcmc_corrected = results["rcmc_result"]

                            # Calculate range-compressed data (for plotting comparison)
                            range_window = np.hamming(raw_data.shape[1])
                            range_fft = np.fft.fft(raw_data * range_window[np.newaxis, :], axis=1)
                            range_mf = create_range_matched_filter(raw_data.shape[1], radar_params)
                            range_compressed_freq = range_fft * range_mf[np.newaxis, :]
                            range_compressed = np.fft.ifft(range_compressed_freq, axis=1)

                            # Plot results up to Step 5
                            fig, axes = plt.subplots(4, 1, figsize=(12, 20))

                            # Plot 1: Raw data
                            raw_db = 20 * np.log10(np.abs(raw_data) + 1e-10)
                            vmin_raw, vmax_raw = np.percentile(raw_db, [5, 98])
                            im1 = axes[0].imshow(raw_db, aspect='auto', cmap='viridis', vmin=vmin_raw, vmax=vmax_raw)
                            axes[0].set_title(f"{radar} - Raw SAR Data")
                            axes[0].set_xlabel("Range Samples")
                            axes[0].set_ylabel("Azimuth Samples")
                            plt.colorbar(im1, ax=axes[0], label="Amplitude (dB)")

                            # Plot 2: Range compressed data
                            range_db = 20 * np.log10(np.abs(range_compressed) + 1e-10)
                            vmin_rc, vmax_rc = np.percentile(range_db, [5, 98])
                            im2 = axes[1].imshow(range_db, aspect='auto', cmap='jet', vmin=vmin_rc, vmax=vmax_rc)
                            axes[1].set_title(f"{radar} - Range Compressed Data")
                            axes[1].set_xlabel("Range Bins")
                            axes[1].set_ylabel("Azimuth Samples")
                            plt.colorbar(im2, ax=axes[1], label="Amplitude (dB)")

                            # Plot 3: After SRC and Range IFFT
                            rd_db = 20 * np.log10(np.abs(range_doppler) + 1e-10)
                            vmin_rd, vmax_rd = np.percentile(rd_db, [5, 98])
                            im3 = axes[2].imshow(rd_db, aspect='auto', cmap='plasma', vmin=vmin_rd, vmax=vmax_rd)
                            axes[2].set_title(f"{radar} - After SRC and Range IFFT")
                            axes[2].set_xlabel("Range Bins")
                            axes[2].set_ylabel("Doppler Bins")
                            plt.colorbar(im3, ax=axes[2], label="Amplitude (dB)")

                            # Plot 4: After RCMC
                            rcmc_db = 20 * np.log10(np.abs(rcmc_corrected) + 1e-10)
                            vmin_rcmc, vmax_rcmc = np.percentile(rcmc_db, [5, 98])
                            im4 = axes[3].imshow(rcmc_db, aspect='auto', cmap='inferno', vmin=vmin_rcmc, vmax=vmax_rcmc)
                            axes[3].set_title(f"{radar} - After RCMC")
                            axes[3].set_xlabel("Range Bins")
                            axes[3].set_ylabel("Doppler Bins")
                            plt.colorbar(im4, ax=axes[3], label="Amplitude (dB)")

                            plt.tight_layout()
                            plt.show()
                            
                        else:
                            print(f"Warning: '{radar}/raw_adc' dataset not found in the HDF5 file")
            
            except Exception as e:
                print(f"Error processing file: {e}")
                import traceback
                traceback.print_exc()
    
    if not found_file:
        print(f"Could not find file with name {scene_name} and any of these extensions: {extensions}")
        print("Please check if the file exists and has the correct name and extension.")

# Execute the main function when script is run directly
if __name__ == "__main__":
    main()