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
                # First, let's print the structure of the HDF5 file to see available datasets
                print("HDF5 file structure:")
                def print_structure(name, obj):
                    print(f"- {name} : {type(obj)}")
                    if isinstance(obj, h5py.Dataset):
                        print(f"  Shape: {obj.shape}, Type: {obj.dtype}")
                data.visititems(print_structure)
                
                # Try to plot the first dataset that looks like an image
                found_dataset = False
                for name, obj in data.items():
                    if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 2:
                        print(f"Attempting to plot dataset: {name}")
                        plt.figure(figsize=(10, 8))
                        plt.imshow(np.abs(obj[()]), cmap="gray")
                        plt.title(f"Dataset: {name}")
                        plt.colorbar(label="Value")
                        plt.show()
                        found_dataset = True
                        break
                
                if not found_dataset:
                    print("No suitable datasets found for plotting")
                
                ##################################
                # PLOTTING
                ##################################
                radar_frequencies = ["80_GHz_Radar", "144_GHz_Radar", "240_GHz_Radar"]
                
                for radar in radar_frequencies:
                    raw_adc = data[f"{radar}/raw_adc"][()]
                    
                    # Create a figure with 2 rows and 2 columns for better visualization
                    plt.figure(figsize=(15, 12))
                    
                    # Plot 1: Raw ADC data magnitude as image
                    plt.subplot(2, 2, 1)
                    raw_magnitude = np.abs(raw_adc)
                    # Use percentile-based scaling to handle outliers
                    vmin, vmax = np.percentile(raw_magnitude, [5, 95])
                    plt.imshow(raw_magnitude, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
                    plt.title(f"{radar} - Raw ADC Data (Magnitude)")
                    plt.xlabel("Range Samples")
                    plt.ylabel("Azimuth Samples")
                    plt.colorbar(label="Amplitude")
                    
                    # Plot 2: Raw ADC data in dB scale
                    plt.subplot(2, 2, 2)
                    raw_db = 20 * np.log10(raw_magnitude + 1e-10)
                    vmin_db, vmax_db = np.percentile(raw_db, [5, 95])
                    plt.imshow(raw_db, aspect='auto', cmap='gray', vmin=vmin_db, vmax=vmax_db)
                    plt.title(f"{radar} - Raw ADC Data (dB)")
                    plt.xlabel("Range Samples")
                    plt.ylabel("Azimuth Samples")
                    plt.colorbar(label="Amplitude (dB)")
                    
                    # Plot 3: Range-compressed data (FFT along range dimension)
                    plt.subplot(2, 2, 3)
                    range_compressed = np.fft.fftshift(np.fft.fft(raw_adc, axis=1), axes=1)
                    range_compressed_db = 20 * np.log10(np.abs(range_compressed) + 1e-10)
                    
                    # Use reasonable dynamic range for visualization
                    vmin_rc, vmax_rc = np.percentile(range_compressed_db, [10, 99.5])
                    plt.imshow(range_compressed_db, aspect='auto', cmap='gray', vmin=vmin_rc, vmax=vmax_rc)
                    plt.title(f"{radar} - Range-Compressed Data (Full)")
                    plt.xlabel("Range Bins")
                    plt.ylabel("Azimuth Samples")
                    plt.colorbar(label="Amplitude (dB)")
                    
                    # Plot 4: Range-compressed data (zoomed to center region)
                    plt.subplot(2, 2, 4)
                    center = range_compressed.shape[1] // 2
                    width = min(512, range_compressed.shape[1] // 4)
                    start_idx = center - width
                    end_idx = center + width
                    
                    plt.imshow(range_compressed_db[:, start_idx:end_idx], 
                              aspect='auto', cmap='gray', vmin=vmin_rc, vmax=vmax_rc)
                    plt.title(f"{radar} - Range-Compressed Data (Zoomed)")
                    plt.xlabel("Range Bins")
                    plt.ylabel("Azimuth Samples")
                    plt.colorbar(label="Amplitude (dB)")
                    
                    plt.tight_layout()
                    plt.show()
                
                # Also plot gantry position information
                plt.figure(figsize=(12, 8))
                plt.subplot(3, 1, 1)
                plt.plot(data["gantry/linear_sample_times"][()], data["gantry/linear_encoder_pos"][()])
                plt.title("Linear Position vs Time")
                plt.xlabel("Time (s)")
                plt.ylabel("Position")
                
                plt.subplot(3, 1, 2)
                plt.plot(data["gantry/roll_sample_times"][()], data["gantry/roll_encoder_pos"][()])
                plt.title("Roll Position vs Time")
                plt.xlabel("Time (s)")
                plt.ylabel("Position")
                
                plt.subplot(3, 1, 3)
                plt.plot(data["gantry/yaw_sample_times"][()], data["gantry/yaw_encoder_pos"][()])
                plt.title("Yaw Position vs Time")
                plt.xlabel("Time (s)")
                plt.ylabel("Position")
                
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Error opening file with extension {ext}: {e}")

if not found_file:
    print(f"Could not find file with name {scene_name} and any of these extensions: {extensions}")
    print("Please check if the file exists and has the correct name and extension.")