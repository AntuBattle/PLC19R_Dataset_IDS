import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Preprocessing
# Load the CSV file
data = pd.read_csv('csv/test_dataset.csv')

# Extract the Voltage column
voltage_values = data['Voltage (V)'].values

# Check for NaN or infinite values and remove them
voltage_values = voltage_values[~np.isnan(voltage_values)]  # Remove NaN
voltage_values = voltage_values[np.isfinite(
    voltage_values)]  # Remove infinite values

# Normalize the entire dataset (before splitting into normal/anomalous)
scaler = MinMaxScaler()
voltage_values_normalized = scaler.fit_transform(voltage_values.reshape(-1, 1))

# Reshape into sequences
sequence_length = 1000  # Choose a sequence length that divides the data evenly
num_sequences = len(voltage_values_normalized) // sequence_length

# Ensure we have enough sequences
if num_sequences == 0:
    raise ValueError(
        "Not enough data to create sequences. Reduce sequence_length or collect more data.")

sequences = voltage_values_normalized[:num_sequences *
                                      sequence_length].reshape(num_sequences, sequence_length, 1)

# Splitting into training and validation sets
split_4_5 = int(0.8 * num_sequences)  # First 4/5 of sequences (normal data)
split_1_5 = int(0.2 * num_sequences)  # Last 1/5 of sequences (anomalies)

# Train set: First 4/5 of sequences
train_sequences = sequences[:split_4_5]

# Validation set: Last 2/5 of sequences (split into normal and anomalies)
validation_sequences = sequences[-(2 * split_1_5):]
# Initially, all validation data is normal
validation_labels = np.zeros(2 * split_1_5)

# Last 1/5 (which is 17/2 of validation set) should be anomalies
anomaly_start_index = int(-split_1_5)
validation_labels[anomaly_start_index:] = 1  # Mark anomalies

# Step 2: Build and Train Autoencoder
input_shape = (sequence_length, 1)
inputs = Input(shape=input_shape)

# Use tanh activation instead of relu
encoded = LSTM(32, activation='tanh')(inputs)
decoded = RepeatVector(sequence_length)(encoded)
decoded = LSTM(32, activation='tanh', return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(1))(decoded)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(
    train_sequences, train_sequences,
    epochs=20,
    batch_size=32,
    validation_data=(validation_sequences, validation_sequences),
    shuffle=True
)

# Step 3: Detect Anomalies
# Reconstruct the sequences
reconstructed_sequences = autoencoder.predict(validation_sequences)

# Compute the reconstruction error (Mean Squared Error)
mse = np.mean(np.square(validation_sequences -
              reconstructed_sequences), axis=(1, 2))

# Define a threshold for anomaly detection
threshold = np.mean(mse) + 2 * np.std(mse)  # Example threshold
predicted_labels = (mse > threshold).astype(
    int)  # 1 for anomalies, 0 for normal

# Step 4: Calculate Confusion Matrix
conf_matrix = confusion_matrix(validation_labels, predicted_labels)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Extract TN, FP, FN, TP from the confusion matrix
if conf_matrix.shape == (1, 1):  # Edge case: all predictions are one class
    TN, FP, FN, TP = conf_matrix[0, 0], 0, 0, 0
elif conf_matrix.shape == (1, 2):  # Edge case: only normal detected
    TN, FP, FN, TP = conf_matrix[0, 0], conf_matrix[0, 1], 0, 0
elif conf_matrix.shape == (2, 1):  # Edge case: only anomalies detected
    TN, FP, FN, TP = 0, 0, conf_matrix[1, 0], conf_matrix[1, 1]
else:
    TN, FP, FN, TP = conf_matrix.flatten()

# Calculate metrics
accuracy = (TN + TP) / (TN + TP + FP + FN) if (TN + TP + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision +
                                       recall) if (precision + recall) > 0 else 0

print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Positives (TP): {TP}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# Step 5: Plot the Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix, display_labels=["Normal", "Anomaly"])
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()
