import pandas as pd

# Load the CSV file from the updated file path
file_path = "/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/labels/transitions_scoresheet_LSTM_3D.csv"
df = pd.read_csv(file_path)

# Function to update the file_name
def update_file_name(file_name):
    # Replace 'RR' with 'MC' and remove the '.mat' extension
    file_name = file_name.replace('_RR', '_MC').replace('.mat', '')
    
    # Split the file name and rearrange the parts
    parts = file_name.split('_')
    tx_part = next((part for part in parts if 'tx' in part), None)
    v_part_index = next((i for i, part in enumerate(parts) if 'V' in part), None)
    
    # If both 'tx' and 'V' parts are found
    if tx_part and v_part_index is not None:
        parts.remove(tx_part)
        # Insert the 'tx' part after 'V' part
        parts.insert(v_part_index + 1, tx_part)
    
    return '_'.join(parts) + '.npy'

# Update the 'file_name' column
df['file_name'] = df['file_name'].apply(update_file_name)

# Generate new rows for additional channels (5-8)
new_rows = []
for _, row in df.iterrows():
    base_name, channel_part = row['file_name'].rsplit('_', 1)
    channel_number = int(channel_part.replace('channel_', '').replace('.npy', ''))
    transition, score = row['transition'], row['score']
    
    # Create new rows for channels 5-8 if the current channel is between 1 and 4
    if channel_number == 4:
        for new_channel in range(5, 9):
            new_file_name = f"{base_name}_{new_channel}.npy"
            new_rows.append({'file_name': new_file_name, 'transition': transition, 'score': score})

# Convert new rows to DataFrame
new_rows_df = pd.DataFrame(new_rows)

# Append the new rows to the original DataFrame
df = pd.concat([df, new_rows_df], ignore_index=True)

# Sort the DataFrame based on the file_name to keep the order consistent
df['sort_value'] = df['file_name'].apply(lambda x: int(x.split('_')[0]))
df.sort_values(by=['sort_value', 'file_name'], inplace=True)
df.drop('sort_value', axis=1, inplace=True)

# Reset the index of the DataFrame
df.reset_index(drop=True, inplace=True)

# Define the output file path
output_file_path = "/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/labels/simulation_labels.csv"

# Write the updated DataFrame to a new CSV file with the header
df.to_csv(output_file_path, index=False)  # Include the header

print("CSV file has been updated and sorted with the new file paths.")
