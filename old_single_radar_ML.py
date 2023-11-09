from keras.models import Sequential
from keras import Model
from keras.layers import Conv3D, ConvLSTM2D, Dense, BatchNormalization, LSTM, Dropout, TimeDistributed, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras_visualizer import visualizer 
import numpy as np
import pandas as pd
import os
import imageio
from time import sleep
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import scipy
import tensorflow as tf
import json

num_classes=6
# Define the top-2 accuracy metric
def top_2_accuracy(y_true, y_pred):
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
    return tf.keras.metrics.top_k_categorical_accuracy(y_true_one_hot, y_pred, k=2)

### load the files
def load_mat_files(data_directory, label_file):
    
    labels_df = pd.read_csv(label_file)

    labels_df = labels_df[(labels_df['transition'] != 'CHAIR') & (labels_df['transition'] != 'YOGIS') & (labels_df['transition'] != 'DOWND') ]

    data_list = []
    scores = []
    transitions = []
    participants = []
    
    for root, dirs, files in os.walk(data_directory):
        for filename in files:
            if filename.endswith('.mat'):
                # Get corresponding row in labels dataframe
                row = labels_df[labels_df['file_name'] == filename]
                
                # If a corresponding row was found in the labels dataframe
                if not row.empty and pd.notna(row['transition'].values[0]):
                    mat = scipy.io.loadmat(os.path.join(root,filename))
                    # The loaded data is a dict with variable names as keys. 
                    # Assuming there's a single variable in the .mat file, get the value
                    var_name = list(mat.keys())[-1]
                    data = mat[var_name]
                    data_list.append(data)
                    if np.array(data).shape != (84,33,32):
                        print(f"File {filename} has shape {np.array(data).shape}")
                    
                    # Extract participant name from the first two characters of the filename
                    participant = filename[:2]
                    participants.append(participant)
                    
                    # Append the score and transition to their respective lists
                    scores.append(row['score'].values[0])
                    transitions.append(row['transition'].values[0])
                else:
                    print(f"No corresponding transition found for file {filename}. Skipping this file.")
                    
    print(np.array(data_list).shape)
    print(np.array(participants).shape, np.array(scores).shape, np.array(transitions).shape)
    return np.array(data_list), np.array(participants), np.array(scores), np.array(transitions)


# Use the function to load the data
data_directory = '/Users/danielcopeland/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Yoga_Study_Simulated_RADAR'
label_file = '/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/labels/simulation_labels.csv'

# Use your function to load the data
data, participants, scores, transitions = load_mat_files(data_directory, label_file)

# Save the variables to the working directory
with open('/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/saved_data_SGD.pkl', 'wb') as f:
    pickle.dump((data, participants, scores, transitions), f)


# #Load the variables
# with open('/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/saved_data_SGD.pkl', 'rb') as f:
#     data, participants, scores, transitions = pickle.load(f)
#     print(np.unique(transitions))


print(len(data))  # prints 3780
print(len(transitions))  # prints 3228

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, transitions, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)

from tensorflow.keras.optimizers import SGD

model = Sequential()

# Convolutional layers for feature extraction
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(None, 33, 32, 1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))  # additional convolution layer
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))  # additional pooling layer
model.add(TimeDistributed(Flatten()))

# LSTM layers for sequence learning
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))  # additional LSTM layer
model.add(LSTM(256))

# Dense layer for classification
model.add(Dense(128, activation='relu'))  # additional Dense layer
model.add(Dense(num_classes, activation='softmax'))

# Create an instance of SGD with a learning rate of 0.01 and momentum of 0.9
sgd = SGD(learning_rate=0.01, momentum=0.9)


# Compile the model with this metric
model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', top_2_accuracy])

model.summary()

visualizer(model,file_name='simple_2_accuracy.pdf', view=True)

# Reshape input to be 5D [samples, timesteps, rows, cols, channels]
X_train = X_train.reshape(X_train.shape[0], 84, 33, 32, 1)
X_test = X_test.reshape(X_test.shape[0], 84, 33, 32, 1)


# Initialize the label encoder
le = LabelEncoder()

# Fit the encoder and transform y_train and y_test
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print(y_train.shape)
print(y_test.shape)
print(np.unique(y_train))
print(np.unique(y_test))

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save it under the form of a json file
json.dump(history.history, open('/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/Year 1/4.453/yoga_project/yoga_python/saved_histories_and_models/transition_top_two_history.json', 'w'))

# Load history
# history = json.load(open('/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/Year 1/4.453/yoga_project/yoga_python/saved_histories_and_models/transition_top_two_history.json', 'r'))

# Save the trained model
model.save('/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/Year 1/4.453/yoga_project/yoga_python/saved_histories_and_models/transition_top_two_model.h5')

import matplotlib.pyplot as plt

# plot the training loss
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Transition Identification Training and Validation Loss') # title here
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot the training accuracy
plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Transition Identification Training and Validation Accuracy') # title here
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# plot the top-2 training accuracy
plt.figure(figsize=(12, 8))
plt.plot(history.history['top_2_accuracy'], label='Training Top-2 Accuracy')
plt.plot(history.history['val_top_2_accuracy'], label='Validation Top-2 Accuracy')
plt.title('Transition Identification Training and Validation Top-2 Accuracy') # title here
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

