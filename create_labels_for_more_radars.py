import pandas as pd

# Load the CSV file into a pandas DataFrame
csv_path = '/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/labels/simulation_labels.csv'  # Update with the path to your CSV file
df = pd.read_csv(csv_path)

# Rename the 'file_name' column to 'base_file_name'
df.rename(columns={'file_name': 'base_file_name'}, inplace=True)

# Number of radar columns to generate
num_radar_columns = 4  # Update this with the number of radar columns you need

# Create the new columns dynamically
for i in range(1, num_radar_columns + 1):
    column_name = f'file_name_{i:03d}'
    df[column_name] = df['base_file_name'].str.replace('.npy', f'_{i:03d}.npy')

df['participant'] = df['base_file_name'].str[:2].astype(str)

# Prepare a list of columns for reordering the DataFrame
reordered_columns = ['participant', 'base_file_name', 'transition', 'score'] + [f'file_name_{i:03d}' for i in range(1, num_radar_columns + 1)]

# Reorder the DataFrame
df = df[reordered_columns]

# Save the updated DataFrame to a new CSV file
df.to_csv('/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/labels/simulation_labels_three_radars.csv', index=False)
