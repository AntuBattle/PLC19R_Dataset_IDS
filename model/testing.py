import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, BatchNormalization, UpSampling1D, AveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sys import argv

# ========================
# CONFIGURATION PARAMETERS
# ========================
COMMAND_LENGTH = 14500     # Points per complete command
SEQUENCE_LENGTH = 14500      # Increased to capture temporal patterns
UNAUTH_SEQUENCE_LENGTH = 14500
OVERLAP = 20               # Sequences overlap for better coverage
BATCH_SIZE = 64            # Smaller batches for better gradient estimates
EPOCHS = 100                # With early stopping
TRAIN_RATIO = 0.85         # Percentage of authorized commands for training
NOISE_FACTOR = 0.02        # Data augmentation noise level

# ========================
# DATA PROCESSING PIPELINE
# ========================

# Load and verify data
data = pd.read_csv(argv[1], header=0)
voltage = data['Voltage (V)'].values

# Verify dataset structure
total_points = 2721232
assert len(voltage) == total_points, f"Data mismatch: Expected {total_points} points, got {len(voltage)}"

# Split into authorized and unauthorized
authorized_data = voltage[:2588441]
unauthorized_data = voltage[2588441:]

def safe_split_commands(data, command_length):
    """Split data into complete commands with exact length"""
    num_commands = len(data) // command_length
    return [data[i*command_length:(i+1)*command_length]
            for i in range(num_commands)]

# Split commands
auth_commands = safe_split_commands(authorized_data, COMMAND_LENGTH)
train_commands = auth_commands[:int(TRAIN_RATIO * len(auth_commands))]
val_commands = auth_commands[len(train_commands):]
unauth_commands = safe_split_commands(unauthorized_data, 14500)

# Create global scaler
train_commands_flat = np.concatenate(train_commands)
scaler = MinMaxScaler().fit(train_commands_flat.reshape(-1, 1))

def create_sequences(commands, seq_length, scaler, overlap=0):
    """Create sequences with overlap using global scaler"""
    sequences = []
    for cmd in commands:
        if len(cmd) < seq_length:
            continue

        # Normalize with global scaler
        normalized_cmd = scaler.transform(cmd.reshape(-1, 1))

        # Create overlapping sequences
        step = max(1, seq_length - overlap)
        num_seq = (len(normalized_cmd) - seq_length) // step + 1
        for i in range(num_seq):
            start = i * step
            seq = normalized_cmd[start:start+seq_length]
            sequences.append(seq)
    return np.array(sequences)

# Generate sequences
train_seq = create_sequences(train_commands, SEQUENCE_LENGTH, scaler, OVERLAP)
val_seq = create_sequences(val_commands, SEQUENCE_LENGTH, scaler, OVERLAP)
test_seq = create_sequences(unauth_commands, UNAUTH_SEQUENCE_LENGTH, scaler, OVERLAP)

# Add noise to training data
train_seq_noisy = train_seq + NOISE_FACTOR * np.random.normal(size=train_seq.shape)
train_seq_noisy = np.clip(train_seq_noisy, 0, 1)

# Prepare labels
val_labels = np.zeros(len(val_seq))
test_labels = np.ones(len(test_seq))

# ========================
# ENHANCED MODEL ARCHITECTURE
# ========================

def build_cnn_lstm_model(seq_len):
    inputs = Input(shape=(14500, 1))

    # CNN Downsampling
    x = Conv1D(32, 10, activation='relu', padding='same')(inputs)
    x = AveragePooling1D(2)(x)  # 7250
    x = Conv1D(64, 10, activation='relu', padding='same')(x)
    x = AveragePooling1D(2)(x)  # 3625

    # LSTM Processing
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64)(x)

    # Decoder
    x = RepeatVector(3625)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(128, return_sequences=True)(x)

    # CNN Upsampling
    x = Conv1D(64, 10, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 10, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)

    outputs = Conv1D(1, 10, activation='sigmoid', padding='same')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_cnn_lstm_model(SEQUENCE_LENGTH)

# ========================
# TRAINING PIPELINE
# ========================

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

history = model.fit(
    train_seq_noisy, train_seq,  # Noisy input, clean target
    validation_data=(val_seq, val_seq),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=False
)

# ========================
# EVALUATION & THRESHOLDING
# ========================

def calculate_mse(sequences):
    reconstructions = model.predict(sequences, batch_size=BATCH_SIZE)
    return np.mean(np.square(sequences - reconstructions), axis=(1, 2))

val_mse = calculate_mse(val_seq)
print(f"Test sequences type: {type(test_seq)}")
print(f"Test sequences shape: {test_seq.shape}")
print(f"First 5 samples sum: {np.sum(test_seq[:5])}")
test_mse = calculate_mse(test_seq)

# Combine all data for threshold analysis
combined_mse = np.concatenate([val_mse, test_mse])
combined_labels = np.concatenate([val_labels, test_labels])

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(combined_labels, combined_mse)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_idx]

# Apply optimal threshold
pred_labels = (combined_mse > optimal_threshold).astype(int)

# Metrics
cm = confusion_matrix(combined_labels, pred_labels)
tn, fp, fn, tp = cm.ravel()

print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"Recall: {tp/(tp+fn):.4f}")
print(f"Precision: {tp/(tp+fp):.4f}")
print(f"F1-Score: {2*(tp/(tp+fp)*(tp/(tp+fn))/(tp/(tp+fp)+tp/(tp+fn))):.4f}")
print("Confusion Matrix:")
print(cm)

# Visualization
plt.figure(figsize=(12, 5))

# Training history
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training History')
plt.legend()

# Error distribution
plt.subplot(1, 2, 2)
plt.hist(val_mse, bins=50, alpha=0.5, label='Authorized')
plt.hist(test_mse, bins=50, alpha=0.5, label='Unauthorized')
plt.axvline(optimal_threshold, color='r', linestyle='--', label='Threshold')
plt.title('Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()
