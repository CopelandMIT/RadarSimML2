import pandas as pd

# Load the CSV file into a pandas DataFrame
csv_path = '/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/labels/simulation_labels.csv'  # Update with the path to your CSV file
df = pd.read_csv(csv_path)

# Rename the 'file_name' column to 'base_file_name'
df.rename(columns={'file_name': 'base_file_name'}, inplace=True)

# Create the new columns for 'file_name_001', 'file_name_002', 'file_name_003' by replacing the file extension
df['file_name_001'] = df['base_file_name'].str.replace('.npy', '_001.npy')
df['file_name_002'] = df['base_file_name'].str.replace('.npy', '_002.npy')
df['file_name_003'] = df['base_file_name'].str.replace('.npy', '_003.npy')
df['participant'] = df['base_file_name'].str[:2].astype(str)

# Reorder the DataFrame
df = df[['participant', 'base_file_name', 'transition', 'score', 'file_name_001', 'file_name_002', 'file_name_003']]

# Save the updated DataFrame to a new CSV file
df.to_csv('/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/labels/simulation_labels_three_radars.csv', index=False)
