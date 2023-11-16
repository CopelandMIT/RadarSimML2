import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, BatchNormalization, Dropout, Concatenate, Input
# from tensorflow.keras.optimizers import SGD #USE for linux of non-M1/M2 mac
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_DIR = '/Users/danielcopeland/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Yoga_Study_Simulated_RADAR'
participants = ['01', '02', '03', '04', '05', '08', '10', '12', '13', '14', '15', '16', '18', '22', '24']

# Load label data from CSV file
labels_csv_path = "/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RadarSimML/labels/simulation_labels_multi_radars.csv"
labels_df = pd.read_csv(labels_csv_path)

# Zero-fill the 'participant' column to a string of length 2
labels_df['participant'] = labels_df['participant'].apply(lambda x: str(x).zfill(2))

# Filter for CRW2L and CRW2R
labels_df = labels_df[(labels_df['transition'] == 'CRW2L') | (labels_df['transition'] == 'CRW2R')]

# # Convert 'score' to numeric values (assuming 'score' is ordinal and already in a suitable numeric format)
labels_df['score'] = pd.to_numeric(labels_df['score'])

# Printing unique participants
unique_participants = labels_df['participant'].unique()
print(unique_participants)

# Create a mapping from file name to numeric score
file_to_score = {row['base_file_name']: row['score'] for _, row in labels_df.iterrows()}

participants_data = {p: {'X': [], 'y': []} for p in participants}  # Dictionary to store data for each participant

for participant in participants:
    participant_data = labels_df[labels_df['participant'].astype(str) == str(participant)]
    print(participant)
    print(participant_data)
    for _, row in participant_data.iterrows():
        videos = []
        for suffix in ['001', '002', '004']:
            file_name = row[f'file_name_{suffix}']
            file_path = os.path.join(DATA_DIR, str(participant), suffix, file_name)
            print(f"Looking for {file_path}")
            
            if os.path.exists(file_path):
                video = np.load(file_path).astype(np.float16)
                videos.append(video)
                print(f'Loaded file {file_path}')
            else:
                print(f'File not found: {file_path}')
        
        if len(videos) == 3:
            participants_data[participant]['X'].append(tuple(videos))
            score = file_to_score[row['base_file_name']]
            participants_data[participant]['y'].append(score)

# Now, each participant's data is stored separately in participants_data

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


# Define the dimensions
m, n, p = 300, 24, 114  # Adjust these based on your data

# Function to reshape the data
def reshape_data(X):
    # Assuming X is a list of tuples (videos), convert it to a NumPy array and reshape
    X_array = np.array(X)
    return X_array.reshape(X_array.shape[0], 3, m, n, p, 1)  # 3 for three videos per entry


input_shape =  (3, m, n, p, 1)  # 3 for three videos per entry


# Store evaluation metrics for each participant
evaluation_metrics = {'participant': [], 'mse': [], 'mae': []}

# Debugging: Print the number of entries for each participant
for participant, data in participants_data.items():
    print(f"Participant {participant} has {len(data['X'])} data entries.")

# Leave-one-out cross-validation loop
for test_participant in participants_data:
    # Preparing training and testing data
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for participant, data in participants_data.items():
        if participant != test_participant:
            X_train.extend(data['X'])
            y_train.extend(data['y'])
        else:
            X_test.extend(data['X'])
            y_test.extend(data['y'])
    
    # Reshape training and testing data
    X_train_reshaped = reshape_data(X_train)
    X_test_reshaped = reshape_data(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Debugging: Print shapes after separation
    print(f"LOOCV for participant {test_participant}:")
    print(f"Training data shape: {X_train_reshaped.shape}, {len(y_train)}")
    print(f"Testing data shape: {X_test_reshaped.shape}, {len(y_test)}")

    # ... rest of the code for training and evaluating the model ...

    
    # Separating the videos in the training set
    X_train_video1 = X_train_reshaped[:, 0, :, :, :, :]
    X_train_video2 = X_train_reshaped[:, 1, :, :, :, :]
    X_train_video3 = X_train_reshaped[:, 2, :, :, :, :]

    # Separating the videos in the testing set
    X_test_video1 = X_test_reshaped[:, 0, :, :, :, :]
    X_test_video2 = X_test_reshaped[:, 1, :, :, :, :]
    X_test_video3 = X_test_reshaped[:, 2, :, :, :, :]
    
    print(f"Training data shape: {X_train_video1.shape}, {X_train_video2.shape}, {X_train_video3.shape}, {len(y_train)}")
    print(f"Testing data shape: {X_test_video1.shape}, {X_test_video2.shape}, {X_test_video3.shape}, {y_test.shape}")
    
    print(f"Check for NaN/Inf in training data: {np.isnan(X_train_video1).any()}, {np.isinf(X_train_video1).any()}")
    print(f"Check for NaN/Inf in testing data: {np.isnan(X_test_video1).any()}, {np.isinf(X_test_video1).any()}")


    # Creating the model
    input_video1 = Input(shape=input_shape[1:])  # Adjust the shape
    input_video2 = Input(shape=input_shape[1:])
    input_video3 = Input(shape=input_shape[1:])

    branch1 = create_branch(input_video1)
    branch2 = create_branch(input_video2)
    branch3 = create_branch(input_video3)

    merged = Concatenate()([branch1, branch2, branch3])

    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=[input_video1, input_video2, input_video3], outputs=output)

    sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)

    model.compile(optimizer=sgd_optimizer, loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])

    model.summary()

    
    try:
        # Training the model
        history = model.fit(
            [X_train_video1, X_train_video2, X_train_video3], 
            np.array(y_train), 
            validation_data=([X_test_video1, X_test_video2, X_test_video3], y_test), 
            epochs=1, 
            batch_size=16
        )
        
        # Evaluate the model
        mse, mae = model.evaluate([X_test_video1, X_test_video2, X_test_video3], y_test)
        # Store the evaluation metrics
        evaluation_metrics['participant'].append(test_participant)
        evaluation_metrics['mse'].append(mse)
        evaluation_metrics['mae'].append(mae)


        # Save the model for each participant
        model.save(f'model_LOO/three_radar_quality_model_{test_participant}')
    except Exception as e:
        print(f"Error occurred with participant {test_participant}: {e}")
    
    
# Visualize the model architecture
plot_model(model, to_file='three_radar_quality_model_LOO.pdf', show_shapes=True)

# Plotting the results
plt.figure(figsize=(10, 5))

# Mean Squared Error Plot
plt.subplot(1, 2, 1)
plt.bar(evaluation_metrics['participant'], evaluation_metrics['mse'])
plt.xlabel('Participant')
plt.ylabel('MSE')
plt.title('Mean Squared Error by Participant')

# Mean Absolute Error Plot
plt.subplot(1, 2, 2)
plt.bar(evaluation_metrics['participant'], evaluation_metrics['mae'])
plt.xlabel('Participant')
plt.ylabel('MAE')
plt.title('Mean Absolute Error by Participant')

plt.tight_layout()
plt.show()