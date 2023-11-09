import os
import numpy as np

# Set the directory path
dir_path = '/Users/danielcopeland/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Yoga_Study_Simulated_RADAR/01/radar_sim_npy/'

# Find the first .npy file in the directory
npy_file = None
for filename in os.listdir(dir_path):
    if filename.endswith('.npy'):
        npy_file = filename
        break  # Stop after finding the first .npy file

# If a .npy file is found, load it and print its shape and contents
if npy_file:
    file_path = os.path.join(dir_path, npy_file)
    data = np.load(file_path)
    
    # Print the shape of the data
    print(f"The shape of the loaded data is: {data.shape}")

    # Print the contents of the data (first few elements)
    print("Contents of the loaded data (first few elements):")
    print(data.flat[:5])  # Using .flat to handle multidimensional arrays safely
else:
    print("No .npy file found in the directory.")
