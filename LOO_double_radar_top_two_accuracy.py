import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, BatchNormalization, Dropout, Concatenate
# from tensorflow.keras.optimizers import SGD #USE for linux of non-M1/M2 mac
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

NUM_CLASSES = 6
DATA_DIR = '/Users/danielcopeland/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Yoga_Study_Simulated_RADAR'
participants = ['01', '02', '03', '04', '05', '08', '10', '12', '13', '14', '15', '16', '18', '22', '24']

# Load label data from CSV file
labels_csv_path = "/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/labels/simulation_labels_multi_radars.csv"
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
video_pairs = []  # This will store pairs of videos
video_labels = []  # This will store the one-hot encoded labels

for participant in participants:
    participant_data = labels_df[labels_df['participant'].astype(str) == participant]
    
    for _, row in participant_data.iterrows():
        videos = []
        for suffix in ['001', '002']:
            file_name = row[f'file_name_{suffix}']
            file_path = os.path.join(DATA_DIR, participant, suffix, file_name)
            
            # Check if the file exists
            if os.path.exists(file_path):
                video = np.load(file_path).astype(np.float16)  # Load the video and cast to float16
                videos.append(video)
                print(f'Loaded {file_path}')
            else:
                print(f'File not found: {file_path}')
        
        if len(videos) == 2:  # Ensure that both videos are present
            video_pairs.append(tuple(videos))
            label = file_to_label[row['file_name_001']]  # Retrieve the label using the file_name_001
            video_labels.append(label)

# Stack all video pair data and labels into NumPy arrays
X_data = np.stack(video_pairs)
y_data = np.stack(video_labels)

# Reshape X_data considering it now contains pairs of videos
# Update with actual dimensions of your video data
m, n, p = 300, 24, 114  # Replace these with actual dimensions
X_data = X_data.reshape(X_data.shape[0], 2, m, n, p, 1)  # 2 for two videos per entry

print(f"X data shape is {X_data.shape}")
print(X_data[:2])  # Prints the first 2 entries (pairs of videos)

print(f"y data shape is {y_data.shape}")
print(y_data[:2])  # Prints the first 2 labels


# Define the top-2 accuracy metric
top_2_accuracy = TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
from keras.layers import Input, Concatenate
from keras.models import Model

# Function to create a branch of the model
def create_branch(input_layer):
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(input_layer)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(256, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = LSTM(256)(x)
    x = Dropout(0.5)(x)
    return x

# Input shape for each video in the pair
input_shape = (300, 24, 114, 1)  # Assuming each video has this shape


# Function to create the model
def create_model(input_shape, num_classes):
    input_video1 = Input(shape=input_shape)
    input_video2 = Input(shape=input_shape)
    
    branch1 = create_branch(input_video1)
    branch2 = create_branch(input_video2)

    merged = Concatenate()([branch1, branch2])
    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[input_video1, input_video2], outputs=output)
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', TopKCategoricalAccuracy(k=2)])
    return model
# Define dimensions and number of classes
m, n, p = 300, 24, 114  # Replace with actual dimensions
NUM_CLASSES = 6
input_shape = (m, n, p, 1)

# Leave-One-Out Cross-Validation
evaluation_metrics = {'participant': [], 'accuracy': [], 'top_2_accuracy': [], 'loss': []}

for test_participant in participants:
    X_train, y_train = [], []
    for participant, videos, labels in zip(participants, video_pairs, video_labels):
        if participant != test_participant:
            X_train.extend(videos)
            y_train.extend(labels)
    
    X_test = np.array([videos for participant, videos in zip(participants, video_pairs) if participant == test_participant])
    y_test = np.array([labels for participant, labels in zip(participants, video_labels) if participant == test_participant])

    # Reshape data
    X_train = np.array(X_train).reshape(-1, 2, m, n, p, 1)
    X_test = X_test.reshape(-1, 2, m, n, p, 1)
    
    # Separate videos for training and testing
    X_train_video1 = X_train[:, 0, :, :, :, :]
    X_train_video2 = X_train[:, 1, :, :, :, :]
    X_test_video1 = X_test[:, 0, :, :, :, :]
    X_test_video2 = X_test[:, 1, :, :, :, :]

    # Create and train the model
    model = create_model(input_shape, NUM_CLASSES)
    history = model.fit([X_train_video1, X_train_video2], np.array(y_train),
                        epochs=10, batch_size=16)

    # Evaluate the model
    loss, accuracy, top_2_accuracy = model.evaluate([X_test_video1, X_test_video2], y_test)
    evaluation_metrics['participant'].append(test_participant)
    evaluation_metrics['accuracy'].append(accuracy)
    evaluation_metrics['top_2_accuracy'].append(top_2_accuracy)
    evaluation_metrics['loss'].append(loss)

    # Optionally, save the model
    model.save(f'LOO_model_transtion_type_{test_participant}.h5')

import matplotlib.pyplot as plt

# Extracting metrics for plotting
participants = evaluation_metrics['participant']
accuracies = evaluation_metrics['accuracy']
top_2_accuracies = evaluation_metrics['top_2_accuracy']
losses = evaluation_metrics['loss']

# Setting up the figure
plt.figure(figsize=(15, 5))

# Plotting Accuracy
plt.subplot(1, 3, 1)
plt.bar(participants, accuracies, color='blue')
plt.xlabel('Participant')
plt.ylabel('Accuracy')
plt.title('Accuracy per Participant')

# Plotting Top-2 Accuracy
plt.subplot(1, 3, 2)
plt.bar(participants, top_2_accuracies, color='green')
plt.xlabel('Participant')
plt.ylabel('Top-2 Accuracy')
plt.title('Top-2 Accuracy per Participant')

# Plotting Loss
plt.subplot(1, 3, 3)
plt.bar(participants, losses, color='red')
plt.xlabel('Participant')
plt.ylabel('Loss')
plt.title('Loss per Participant')

plt.tight_layout()
plt.show()

