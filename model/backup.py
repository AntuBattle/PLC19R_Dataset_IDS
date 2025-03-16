import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sys import argv, exit

# Defining Macros
TRAIN_SET_SPLIT = 0.75  # Percentage of dataset that will form the training set

# Step 1: Preprocessing
# Load the CSV file
data = pd.read_csv(argv[1])

# Extract the Voltage column
voltage_values = data['Voltage (V)'].values

# Check for NaN or infinite values and remove them
voltage_values = voltage_values[~np.isnan(voltage_values)]  # Remove NaN
voltage_values = voltage_values[np.isfinite(voltage_values)]  # Remove infinite values

# Normalize the entire dataset (before splitting into normal/anomalous)
scaler = MinMaxScaler()
voltage_values_normalized = scaler.fit_transform(voltage_values.reshape(-1, 1))

# Reshape into sequences
sequence_length = 50  # Choose a sequence length that divides the data evenly
num_sequences = len(voltage_values_normalized) // sequence_length

# Ensure we have enough sequences
if num_sequences == 0:
    raise ValueError(
        "Not enough data to create sequences. Reduce sequence_length or collect more data.")

sequences = voltage_values_normalized[:num_sequences *
                                      sequence_length].reshape(num_sequences, sequence_length, 1)

# Splitting into training, validation, and test sets
train_split = int(TRAIN_SET_SPLIT * num_sequences)  # Training set is 75% of the whole dataset
remaining_sequences = sequences[train_split:]  # Remaining 25%

# Split remaining into validation (75% of remaining) and test (25% of remaining)
val_test_split = int(0.75 * len(remaining_sequences))
validation_sequences = remaining_sequences[:val_test_split]
test_sequences = remaining_sequences[val_test_split:]

# Create test labels: half authorized (0), half unauthorized (1)
test_labels = np.zeros(len(test_sequences))
half_test = len(test_sequences) // 3
test_labels[half_test:] = 1  # Mark the latter half as unauthorized

# Train set
train_sequences = sequences[:train_split]

# Step 2: Build and Train Autoencoder
input_shape = (sequence_length, 1)
inputs = Input(shape=input_shape)

# Use tanh activation instead of relu
encoded = LSTM(64, activation='tanh')(inputs)
decoded = RepeatVector(sequence_length)(encoded)
decoded = LSTM(64, activation='tanh', return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(1))(decoded)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(
    train_sequences, train_sequences,
    epochs=200,
    batch_size=64,
    validation_data=(validation_sequences, validation_sequences),
    shuffle=False
)

# Step 3: Detect Anomalies on Test Set
# Reconstruct the test sequences
reconstructed_test = autoencoder.predict(test_sequences)

# Compute the reconstruction error (Mean Squared Error)
mse = np.mean(np.square(test_sequences - reconstructed_test), axis=(1, 2))

# Define a threshold for anomaly detection
threshold = np.mean(mse) + 0.4 * np.std(mse)
predicted_labels = (mse > threshold).astype(int)  # 1 for anomalies, 0 for normal

# Step 4: Calculate Confusion Matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

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
