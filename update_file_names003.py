import os

# Define the base directory and participants
base_dir = "/Users/danielcopeland/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Yoga_Study_Simulated_RADAR"
participants = ['01', '02', '03', '04', '05', '08', '09', '10', '12', '13', '14', '15', '16', '18', '22', '24']

# Loop over each participant
for participant in participants:
    participant_folder = os.path.join(base_dir, participant, '003')
    
    # Check if the participant folder exists
    if not os.path.exists(participant_folder):
        print(f"Folder does not exist: {participant_folder}")
        continue
    
    # Loop over each file in the participant's folder
    for filename in os.listdir(participant_folder):
        # Check if the file contains "002"
        if "002" in filename:
            # Create the new filename by replacing "002" with "003"
            new_filename = filename.replace("002", "003")
            old_file_path = os.path.join(participant_folder, filename)
            new_file_path = os.path.join(participant_folder, new_filename)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {old_file_path} to {new_file_path}")
