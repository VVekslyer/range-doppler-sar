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

# Try different file extensions
extensions = [".h5", ".hdf5", ".hdf"]
found_file = False

for ext in extensions:
    data_name = os.path.join(sar_scene_folder, scene_name + ext)
    if os.path.exists(data_name):
        found_file = True
        print(f"Found file: {data_name}")
        
        try:
            with h5py.File(data_name, "r") as data:
                # First, let's print the structure of the HDF5 file
                print("HDF5 file structure:")
                def print_structure(name, obj):
                    print(f"- {name} : {type(obj)}")
                    if isinstance(obj, h5py.Dataset):
                        print(f"  Shape: {obj.shape}, Type: {obj.dtype}")
                data.visititems(print_structure)
                
                ##################################
                # SIMPLIFIED PLOTTING
                ##################################
                radar_frequencies = ["80_GHz_Radar", "144_GHz_Radar", "240_GHz_Radar"]
                
                for radar in radar_frequencies:
                    raw_adc = data[f"{radar}/raw_adc"][()]
                    
                    # Create a figure with 2 rows and 1 column (stacked vertically)
                    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
                    
                    # Plot 1: Raw ADC data magnitude (dB scale)
                    raw_magnitude = np.abs(raw_adc)
                    raw_db = 20 * np.log10(raw_magnitude + 1e-10)
                    vmin_db, vmax_db = np.percentile(raw_db, [5, 95])
                    im1 = axs[0].imshow(raw_db, aspect='auto', cmap='viridis', vmin=vmin_db, vmax=vmax_db)
                    axs[0].set_title(f"{radar} - Raw ADC Data")
                    axs[0].set_xlabel("Range Samples")
                    axs[0].set_ylabel("Azimuth Samples")
                    fig.colorbar(im1, ax=axs[0], label="Amplitude (dB)")
                    
                    # Plot 2: Range-compressed data (single-sided)
                    # Perform range compression using FFT
                    range_compressed = np.fft.fft(raw_adc, axis=1)
                    
                    # Take only the positive frequency side (first half)
                    n_range = range_compressed.shape[1]
                    single_sided = range_compressed[:, :n_range//2]
                    
                    # Convert to dB scale for better visualization
                    single_sided_db = 20 * np.log10(np.abs(single_sided) + 1e-10)
                    
                    # Use reasonable dynamic range for visualization
                    vmin_rc, vmax_rc = np.percentile(single_sided_db, [10, 99.5])
                    im2 = axs[1].imshow(single_sided_db, aspect='auto', cmap='jet', vmin=vmin_rc, vmax=vmax_rc)
                    axs[1].set_title(f"{radar} - Range-Compressed Data")
                    axs[1].set_xlabel("Range Bins")
                    axs[1].set_ylabel("Azimuth Samples")
                    fig.colorbar(im2, ax=axs[1], label="Amplitude (dB)")
                    
                    plt.tight_layout()
                    plt.show()
                
        except Exception as e:
            print(f"Error opening file with extension {ext}: {e}")

if not found_file:
    print(f"Could not find file with name {scene_name} and any of these extensions: {extensions}")
    print("Please check if the file exists and has the correct name and extension.")