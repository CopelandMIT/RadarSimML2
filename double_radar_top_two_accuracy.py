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
participants = ['01', '02', '03', '04', '05', '08', '09', '10', '12', '13', '14', '15', '16', '18', '22', '24']

# Load label data from CSV file
labels_csv_path = "/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/labels/simulation_labels_multi_radars.csv"
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

# Input shape for each video in the pair
input_shape = (300, 24, 114, 1)  # Assuming each video has this shape

# Define two input layers, one for each video in the pair
input_video1 = Input(shape=input_shape)
input_video2 = Input(shape=input_shape)

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

# Create branches for each input
branch1 = create_branch(input_video1)
branch2 = create_branch(input_video2)

# Merge the branches
merged = Concatenate()([branch1, branch2])

# Final layers for classification
x = Dense(128, activation='relu')(merged)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the model
model = Model(inputs=[input_video1, input_video2], outputs=output)

# Create an instance of SGD with a learning rate of 0.01 and momentum of 0.9
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)

# Compile the model
model.compile(optimizer=sgd_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy', top_2_accuracy])

# Model summary
model.summary()

# Visualize the model architecture
plot_model(model, to_file='2_radar_2_accuracy.pdf', show_shapes=True)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Separate the videos in each tuple
X_train_video1 = np.array([pair[0] for pair in X_train])
X_train_video2 = np.array([pair[1] for pair in X_train])
X_val_video1 = np.array([pair[0] for pair in X_val])
X_val_video2 = np.array([pair[1] for pair in X_val])

# Reshape the arrays
X_train_video1 = X_train_video1.reshape(X_train_video1.shape[0], 300, 24, 114, 1)
X_train_video2 = X_train_video2.reshape(X_train_video2.shape[0], 300, 24, 114, 1)
X_val_video1 = X_val_video1.reshape(X_val_video1.shape[0], 300, 24, 114, 1)
X_val_video2 = X_val_video2.reshape(X_val_video2.shape[0], 300, 24, 114, 1)

print(f"X_train_video1 shape: {X_train_video1.shape}")
print(f"X_train_video2 shape: {X_train_video2.shape}")
print(f"X_val_video1 shape: {X_val_video1.shape}")
print(f"X_val_video2 shape: {X_val_video2.shape}")

# Training the model with history tracking
history = model.fit(
    [X_train_video1, X_train_video2], 
    y_train, 
    validation_data=([X_val_video1, X_val_video2], y_val), 
    epochs=10, 
    batch_size=16
)

# Save the model
model.save('two_radar_yoga_model')

# Plotting the training accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plotting the top-2 accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['top_2_accuracy'], label='Training Top-2 Accuracy')
plt.plot(history.history['val_top_2_accuracy'], label='Validation Top-2 Accuracy')
plt.title('Training and Validation Top-2 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Top-2 Accuracy')
plt.legend()

# Plotting the loss
plt.subplot(1, 3, 3)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

