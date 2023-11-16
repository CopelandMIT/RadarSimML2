import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

NUM_CLASSES = 6
DATA_DIR = '/Users/danielcopeland/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Yoga_Study_Simulated_RADAR'
participants = ['01', '02', '03', '04', '05', '08', '10', '12', '13', '14', '15', '16', '18', '22', '24']

# Load label data from CSV file
labels_csv_path = "/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/labels/simulation_labels_three_radars.csv"
labels_df = pd.read_csv(labels_csv_path)

# Zero-fill the 'participant' column to a string of length 2
labels_df['participant'] = labels_df['participant'].apply(lambda x: str(x).zfill(2))

# Filter out unwanted transitions
labels_df = labels_df[(labels_df['transition'] != 'CHAIR') & (labels_df['transition'] != 'YOGIS') & (labels_df['transition'] != 'DOWND')]

# Convert class labels to integers and then to one-hot encoding
labels_df['transition_code'] = labels_df['transition'].astype('category').cat.codes
labels_one_hot = to_categorical(labels_df['transition_code'].values, num_classes=NUM_CLASSES)

# Create a mapping from file name to one-hot encoded label
file_to_label = {row['file_name_001']: to_categorical(row['transition_code'], num_classes=NUM_CLASSES)
                 for _, row in labels_df.iterrows()}

# Load the video data and pair it with labels
video_data = []
video_labels = []  # This will store the one-hot encoded labels

for participant in participants:
    # Filter the DataFrame for the current participant
    participant_data = labels_df[labels_df['participant'].astype(str) == participant]
    
    for _, row in participant_data.iterrows():
        file_path = os.path.join(DATA_DIR, participant, '001', row['file_name_001'])
        
        # Check if the file exists
        if os.path.exists(file_path):
            video = np.load(file_path).astype(np.float16)  # Load the video and cast to float16
            video_data.append(video)
            label = file_to_label[row['file_name_001']]  # Retrieve the label using the full file name
            video_labels.append(label)
            print(f'Loaded {file_path}')
        else:
            print(f'File not found: {file_path}')
        
# Now video_data contains all the loaded videos in float16, and video_labels contains the corresponding one-hot labels


# Stack all video data into a single NumPy array and convert labels to an array
X_data = np.stack(video_data)
y_data = np.stack(video_labels)

X_data = X_data.reshape(X_data.shape[0], 300, 24, 114, 1)
print(f"X data shape is {X_data.shape}")
print(X_data[:2])  # Prints the first 5 elements of the array

print(f"X data shape is {X_data.shape}")
print(X_data[:2])  # Prints the first 5 elements of the array

print(f"y data shape is {y_data.shape}")
print(y_data[:2])  # Prints the first 5 elements of the array

def create_base_model(frame_shape):
    # frame_shape should be (Height, Width, Channels)
    input_layer = Input(shape=(None, *frame_shape))  # None is for the time steps which will be inferred
    x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))(input_layer)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(units=50, return_sequences=True)(x)
    x = LSTM(units=50)(x)
    return Model(inputs=input_layer, outputs=x)

# Define the input shape with the time steps
input_shape_with_time = (300, 24, 114, 1)  # (Time steps, Height, Width, Channels)

# Create the model with the correct input shape
input_layer = Input(shape=input_shape_with_time)
features = create_base_model(input_shape_with_time[1:])(input_layer)  # Pass the frame shape without the time steps

classification = Dense(units=NUM_CLASSES, activation='softmax')(features)
model = Model(inputs=input_layer, outputs=classification)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 300, 24, 114, 1)
X_val = X_val.reshape(X_val.shape[0], 300, 24, 114, 1)

print(f"X_train shape: {X_train.shape}")  # Should be (None, 300, 24, 114, 1)
print(f"X_val shape: {X_val.shape}")  # Should be (None, 300, 24, 114, 1)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model if needed
model.save('radar_yoga_model.h5')
