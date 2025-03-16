import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sys import argv

# ========================
# CONFIGURATION PARAMETERS
# ========================
COMMAND_LENGTH = 14000     # Points per complete command
SEQUENCE_LENGTH = 20     # Reduced for initial testing
BATCH_SIZE = 128           # Start with smaller batches
EPOCHS = 200               # With early stopping
TRAIN_RATIO = 0.85         # Percentage of authorized commands for training

# ========================
# DATA PROCESSING PIPELINE
# ========================

# Load and verify data
data = pd.read_csv(argv[1], header=0)
voltage = data['Voltage (V)'].values

# Verify dataset structure
total_points = 2721232  # Authorized + Unauthorized
assert len(voltage) == total_points, f"Data mismatch: Expected {total_points} points, got {len(voltage)}"

# Split into authorized and unauthorized
authorized_data = voltage[:2588441]
unauthorized_data = voltage[2588441:]

def safe_split_commands(data, command_length):
    """Split data into complete commands with exact length"""
    num_commands = len(data) // command_length
    return [data[i*command_length:(i+1)*command_length]
            for i in range(num_commands)]

# Split authorized commands
auth_commands = safe_split_commands(authorized_data, COMMAND_LENGTH)
total_auth_commands = len(auth_commands)
train_commands = auth_commands[:int(TRAIN_RATIO * total_auth_commands)]
val_commands = auth_commands[len(train_commands):]

# Split unauthorized commands
unauth_commands = safe_split_commands(unauthorized_data, COMMAND_LENGTH)

# Verify command counts
assert len(auth_commands) > 0, "No complete authorized commands found!"
assert len(unauth_commands) > 0, "No complete unauthorized commands found!"

def create_safe_sequences(commands, seq_length):
    """Create sequences with data validation"""
    sequences = []
    for cmd in commands:
        if len(cmd) < seq_length:
            continue  # Skip commands shorter than sequence length

        # Normalize per command
        cmd_scaler = MinMaxScaler()
        normalized_cmd = cmd_scaler.fit_transform(cmd.reshape(-1, 1))

        # Create non-overlapping sequences
        num_seq = len(normalized_cmd) // seq_length
        for i in range(num_seq):
            seq = normalized_cmd[i*seq_length:(i+1)*seq_length]
            sequences.append(seq)
    return np.array(sequences)

# Create datasets with validation
train_seq = create_safe_sequences(train_commands, SEQUENCE_LENGTH)
val_seq = create_safe_sequences(val_commands, SEQUENCE_LENGTH)
test_seq = create_safe_sequences(unauth_commands, SEQUENCE_LENGTH)

# Final validation checks
assert len(train_seq) > 0, "No training sequences! Reduce SEQUENCE_LENGTH"
assert len(val_seq) > 0, "No validation sequences! Reduce SEQUENCE_LENGTH"
assert len(test_seq) > 0, "No test sequences! Reduce SEQUENCE_LENGTH"

# Create labels (0 = authorized, 1 = unauthorized)
val_labels = np.zeros(len(val_seq))
test_labels = np.ones(len(test_seq))

# Combine validation and test for evaluation
full_test_seq = np.concatenate([val_seq, test_seq])
full_test_labels = np.concatenate([val_labels, test_labels])

# ========================
# MODEL ARCHITECTURE
# ========================

def build_robust_model(seq_len):
    inputs = Input(shape=(seq_len, 1))

    # Encoder
    x = Bidirectional(LSTM(256, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(128)(x)

    # Bottleneck
    encoded = Dense(64, activation='tanh')(x)

    # Decoder
    x = RepeatVector(seq_len)(encoded)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    outputs = TimeDistributed(Dense(1))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

model = build_robust_model(SEQUENCE_LENGTH)

# ========================
# TRAINING & EVALUATION
# ========================

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(
    train_seq, train_seq,
    validation_data=(val_seq, val_seq),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True
)

# Calculate dynamic threshold
val_recon = model.predict(val_seq, batch_size=BATCH_SIZE)
val_mse = np.mean(np.square(val_seq - val_recon), axis=(1, 2))
threshold = np.percentile(val_mse, 99.5)

# Predict and evaluate
test_recon = model.predict(full_test_seq, batch_size=BATCH_SIZE)
test_mse = np.mean(np.square(full_test_seq - test_recon), axis=(1, 2))
pred_labels = (test_mse > threshold).astype(int)

# Metrics
cm = confusion_matrix(full_test_labels, pred_labels)
tn, fp, fn, tp = cm.ravel()

print(f"Recall: {tp/(tp+fn):.4f}")
print(f"Precision: {tp/(tp+fp):.4f}")
print("Confusion Matrix:")
print(cm)

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
