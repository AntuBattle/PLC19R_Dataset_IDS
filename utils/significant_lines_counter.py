import pandas as pd
from sys import argv


data = pd.read_csv(argv[1], dtype={'Time (S)': float, 'Voltage (V)': float})
# Ensure the "Voltage (V)" column is numeric
data['Voltage (V)'] = pd.to_numeric(data['Voltage (V)'], errors='coerce')

# Now apply the filtering
filtered_data = data[(data["Voltage (V)"] > 1.5) |
                     (data["Voltage (V)"] < -1.5)]

print(len(filtered_data))
