"""
This script aims at concatenating more signals to create a variable length dataset.
Signals can be simply concatenated or they can be sampled, reading one value for every n lines, resulting in a lighter, less granular dataset.
"""

import csv
import glob
import os
from sys import argv, exit
from time import sleep

THRESHOLD = 0.2

def print_usage():
    usage = f"""
        Usage: filter_and_concat.py FOLDER_PATH OUT_FILE [STEP]

        This python scipt concatenates every .csv file present into the given FOLDER_PATH, filtering every value > abs({THRESHOLD})
        looking for the "Voltage (V)" column and concatenanting every file into one single output file containing the Voltage (V) column\n of every input file appended one after the other.
        If STEP is given, then the script will read a value will skip the next STEP values. This is used to reduce the size of the output file, depending on the hardware on which the machine is trained on.
        """
    print(usage)


def process_csv_files(input_folder, output_file, steps=1):

    # Get all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    with open(output_file, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['Voltage (V)'])  # Write header

        for file_path in csv_files:
            with open(file_path, 'r') as in_f:
                reader = csv.reader(in_f)
                try:
                    header = next(reader)
                    voltage_index = header.index('Voltage (V)')
                except (StopIteration, ValueError):
                    print(f"Skipping {file_path} - missing header or voltage column")
                    continue

                row_counter = 0
                for row in reader:
                    # Check if we're at every 20th row
                    if row_counter % steps == 0:
                        try:
                            voltage = float(row[voltage_index])
                            if voltage > THRESHOLD or voltage < - (THRESHOLD):
                                writer.writerow([voltage])
                        except (IndexError, ValueError):
                            print(f"Row {row_counter} presented an error or was not porperly formatted. Skipping...")
                            pass  # Skip invalid rows
                    row_counter += 1

if __name__ == '__main__':

    args = len(argv)

    if args < 3 or argv[1] == "-h" or argv[1] == "--help":
        print_usage()
        exit(1)

    input_folder = str(argv[1])
    output_file = str(argv[2])
    steps = 1

    if args >= 4:
        steps = int(argv[3])

    process_csv_files(input_folder, output_file, steps)
    print(f"Processing complete. Output saved to {output_file}")
