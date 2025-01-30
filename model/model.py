import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Preprocessing
# Load the CSV file
data = pd.read_csv('csv/waveform_data_read_coils_combined.csv')

# Extract the Voltage column
voltage_values = data['Voltage (V)'].values

# Check for NaN or infinite values and remove them
voltage_values = voltage_values[~np.isnan(voltage_values)]  # Remove NaN
voltage_values = voltage_values[np.isfinite(voltage_values)]  # Remove infinite values

# Filter significant values (greater than 1.3 or smaller than -1.3)
significant_values = voltage_values[(voltage_values > 1.3) | (voltage_values < -1.3)]

# Ensure significant_values is not empty
if len(significant_values) == 0:
    raise ValueError("No significant voltage values found after filtering. Adjust the threshold.")

# Normalize the significant values
scaler = MinMaxScaler()
significant_values_normalized = scaler.fit_transform(significant_values.reshape(-1, 1))

# Reshape into sequences
sequence_length = 100  # Choose a sequence length that divides the data evenly
num_sequences = len(significant_values_normalized) // sequence_length

# Ensure num_sequences is not zero
if num_sequences == 0:
    raise ValueError("Not enough data to create sequences. Reduce sequence_length or collect more data.")

sequences = significant_values_normalized[:num_sequences * sequence_length].reshape(num_sequences, sequence_length, 1)

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

# Train the autoencoder (enable shuffling for better generalization)
history = autoencoder.fit(
    sequences, sequences,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=True  # Enable shuffling for better learning
)

# Step 3: Detect Anomalies
# Reconstruct the sequences
reconstructed_sequences = autoencoder.predict(sequences)

# Compute the reconstruction error (Mean Squared Error)
mse = np.mean(np.square(sequences - reconstructed_sequences), axis=1)

# Define a threshold for anomaly detection
threshold = np.mean(mse) + 2 * np.std(mse)  # Example threshold
predicted_labels = (mse > threshold).astype(int)  # 1 for anomalies, 0 for normal

# Set all ground truth labels to 0 (normal/authorized)
ground_truth_labels = np.zeros_like(predicted_labels)  # All signals are authorized

# Step 4: Calculate Confusion Matrix
conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Extract TN, FP from the confusion matrix
if conf_matrix.shape == (1, 1):  # Edge case: all predictions are 0
    TN, FP = conf_matrix[0, 0], 0
elif conf_matrix.shape == (1, 2):  # Edge case: only normal detected
    TN, FP = conf_matrix[0, 0], conf_matrix[0, 1]
else:
    TN, FP = conf_matrix.ravel()

# Calculate metrics
accuracy = TN / (TN + FP) if (TN + FP) > 0 else 0
precision = TN / (TN + FP) if (TN + FP) > 0 else 0
recall = 1.0  # Since all ground truth labels are normal, recall is always 1
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# Step 5: Plot the Confusion Matrix
# Use ConfusionMatrixDisplay to plot the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Normal", "Anomaly"])
disp.plot(cmap=plt.cm.Blues, values_format='d')  # Use a blue color map and integer formatting
plt.title("Confusion Matrix")
plt.show()

