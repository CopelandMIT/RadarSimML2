import os
import numpy as np

# Define the function to process .npy files in a folder
def process_npy_files(input_folder, output_folder):
    # Get a list of all .npy files in the input folder
    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    for file_name in npy_files:
        # Load the .npy file from the input folder
        input_file_path = os.path.join(input_folder, file_name)
        data = np.load(input_file_path)
        
        # Get the channel index from the file name (assuming the file name format is "data_channel_X.npy")
        channel_idx = int(file_name.split('_channel_')[1].split('.npy')[0])
        
        # Extract the specified channel and convert to float32
        selected_channel_data = data[:, :, :, channel_idx - 1].astype(np.float32)
        
        # Define a new file name for the modified data
        new_file_name = file_name.replace('.npy', '_001.npy'.format(channel_idx))
        output_file_path = os.path.join(output_folder, new_file_name)
        
        # Save the modified data as a new .npy file in the output folder
        np.save(output_file_path, selected_channel_data)
        print(f'Saved {output_file_path}')

# Define the main function to process all folders and subfolders in the input directory
def process_all_folders(input_dir, output_dir):
    for root, _, _ in os.walk(input_dir):
        if root.endswith('radar_sim_npy'):
            # Create the corresponding output folder structure under the output directory
            relative_path = os.path.relpath(root, input_dir)
            output_folder = os.path.join(output_dir, relative_path)
            os.makedirs(output_folder, exist_ok=True)
            
            # Process the .npy files in the input folder and save them in the corresponding output folder
            process_npy_files(root, output_folder)

# Specify the input and output directories
input_directory = '/Users/danielcopeland/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Yoga_Study_MOCAP_Data'
output_directory = '/Users/danielcopeland/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Yoga_Study_Simulated_RADAR/'

# Call the main function to process all folders and subfolders
process_all_folders(input_directory, output_directory)
