import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, BatchNormalization, Dropout
# from tensorflow.keras.optimizers import SGD #USE for linux of non-M1/M2 mac
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

NUM_CLASSES = 6
DATA_DIR = '/Users/danielcopeland/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Yoga_Study_Simulated_RADAR'
participants = ['01', '02', '03', '04', '05', '08', '09', '10', '12', '13', '14', '15', '16', '18', '22', '24']

# Load label data from CSV file
labels_csv_path = "/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/labels/simulation_labels_three_radars.csv"
labels_df = pd.read_csv(labels_csv_path)

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

# Define the top-2 accuracy metric
top_2_accuracy = TopKCategoricalAccuracy(k=2, name='top_2_accuracy')

# Define the Sequential model
model = Sequential()

# Convolutional layers for feature extraction
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(300, 24, 114, 1)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))

# LSTM layers for sequence learning
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(256, return_sequences=True))  # Additional LSTM layer
model.add(Dropout(0.5))
model.add(LSTM(256))
model.add(Dropout(0.5))

# Dense layer for classification
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))
# Create an instance of SGD with a learning rate of 0.01 and momentum of 0.9
sgd = SGD(learning_rate=0.01, momentum=0.9)

# Compile the model
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy', top_2_accuracy])

# Print the model summary
model.summary()

# Visualize the model architecture
plot_model(model, to_file='simple_2_accuracy.pdf', show_shapes=True)
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
