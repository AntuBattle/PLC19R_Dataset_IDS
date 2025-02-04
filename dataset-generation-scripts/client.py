from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian
import random
from time import sleep

# Configuration
SERVER_IP = "10.10.10.20"
SERVER_PORT = 50222


def main():
    # Connect to the Modbus server
    client = ModbusTcpClient(SERVER_IP, port=SERVER_PORT)
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
                    pass
                """
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
                """
            except Exception as e:
                print(f"An error occurred: {e}")
            sleep(1)

    finally:
        # Close the client connection
        client.close()


if __name__ == "__main__":
    main()
