from pymso5000.mso5000 import MSO5000
from sys import exit, argv
from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian
import random
from time import sleep

MODBUS_SERVER_IP = "10.10.10.20"
MODBUS_SERVER_PORT = 50222

def print_usage():
        print("Usage: python3 oscilloscope.py [OPTIONS] DEST_IP_ADDRESS")
        print("")
        print("OPTIONS")
        print(" -h  --help  Prints a guide on how to use this command.")
        print("")
        print("")
        print("Currently there are no other options implemented.")

try:
    if argv[1] == "-h" or argv[1] == "--help":
        print_usage()
        exit(0)
    else:
        DEST_IP_ADDRESS = argv[1]
except Exception as e:
    print("Invalid or missing argument.")
    print_usage()
    exit()


def send_random_request(client: ModbusTcpClient):
    # Connect to the Modbus server
    client = ModbusTcpClient(MODBUS_SERVER_IP, port=MODBUS_SERVER_PORT)
    if not client.connect():
        print("Failed to connect to Modbus server!")
        return
    try:
        for i in range(10):
            try:
                # Example 1: Read holding registers
                print("Reading holding registers...")
                response = client.read_holding_registers(address=0)
                if response.isError():
                    print("Error reading holding registers:", response)
                else:
                    print("Holding Registers:", response.registers)

                # Example 2: Write a single holding register
                print("\nWriting to a holding register...")
                write_response = client.write_register(address=0, value=random.randint(1, 2000))
                if write_response.isError():
                    print("Error writing to holding register:", write_response)
                else:
                    print("Successfully wrote to holding register.")

                # Example 3: Read discrete inputs
                print("\nReading discrete inputs...")
                discrete_response = client.read_discrete_inputs(address=0, count=5)
                if discrete_response.isError():
                    print("Error reading discrete inputs:", discrete_response)
                else:
                    print("Discrete Inputs:", discrete_response.bits)

                # Example 4: Reading coils
                print("\nReading coils...")
                coil_response = client.read_coils(1)
                if coil_response.isError():
                    print("Error reading Coils:", coil_response)
                else:
                    print("Coils: ", coil_response.bits[0], end="")

                # Example 5: Writing coils
                print("\n Writing Coils...")
                write_coil_response = client.write_coil(1, random.choice([True, False]), slave=1)
                if write_coil_response.isError():
                    print("Error writing coils:", write_coil_response)
                else:
                    print("Successfully wrote Coils.")
            except Exception as e:
                print(f"An error occurred: {e}")
            sleep(1)
            
    finally:
        # Close the client connection
        client.close()




with MSO5000(address = argv[1]) as mso:
    
    print(f"Identify: {mso.identify()}")

    mso.set_channel_enable(0, True)
    # mso.set_channel_enable(1, True)

    mso._scpi.scpiCommand(":STOP")
    mso._scpi.scpiCommand(":CLE")
    mso._scpi.scpiCommand(":TRIG:MODE EDGE")
    mso._scpi.scpiCommand(":TRIG:EDGE:SOUR CHAN1")
    mso._scpi.scpiCommand(":TRIG:EDGE:SLOP POS")
    mso._scpi.scpiCommand(":TRIG:LEV 0.3")

    client = ModbusTcpClient(MODBUS_SERVER_IP, port=MODBUS_SERVER_PORT)
    if not client.connect():
        print("Failed to connect to Modbus server!")
        exit()
    print("Reading holding registers...")
    mso._scpi.scpiCommand(":RUN")

    ### Mando pacchetto
    response = client.read_holding_registers(address=0)
    if response.isError():
        print("Error reading holding registers:", response)
    else:
        print("Holding Registers:", response.registers)

    data = mso._query_waveform(channel=channel)
    mso._scpi.scpiCommand(":STOP")

    # Save data to CSV
    with open('waveform_data.csv', 'w', newline='') as file:
        import csv
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "Voltage (V)"])
        for x, y in zip(data['x'], data['y']):
            writer.writerow([x, y])

    print("Waveform data saved to waveform_data.csv.")
