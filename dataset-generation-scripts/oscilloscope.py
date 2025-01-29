from pymso5000.mso5000 import MSO5000
from labdevices.oscilloscope import OscilloscopeSweepMode, OscilloscopeTriggerMode, OscilloscopeTimebaseMode, OscilloscopeRunMode
from sys import exit, argv
from pymodbus.client import ModbusTcpClient
import random
from time import sleep
from clientHandler import ModbusClientHandler



MODBUS_SERVER_IP = "10.10.11.30"
MODBUS_SERVER_PORT = 50222


def print_usage():
    print("Usage: python3 oscilloscope.py [OPTIONS] OSCILLOSCOPE_IP_ADDRESS")
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


def execute_modbus_command(handler: ModbusClientHandler, command_name: str, mso, file_number: int, **kwargs):
    """
    Executes a Modbus command dynamically using the ModbusClientHandler class.

    :param handler: Instance of ModbusClientHandler.
    :param command_name: Name of the Modbus command to execute (e.g., "read_holding_registers").
    :param mso: Oscilloscope instance for handling SCPI commands.
    :param file_number: File identifier for saving waveform data.
    :param kwargs: Additional parameters to pass to the Modbus command.
    """
    print(f"\nExecuting command: {command_name}...")
    mso._scpi.scpiCommand(":RUN")

    # Allow the line to clear from previous commands
    sleep(2)

    # Execute the Modbus command dynamically
    response = handler.execute_command(command_name, **kwargs)
    print(f"Command response: {response}")

    mso._scpi.scpiCommand(":STOP")
    data = mso._query_waveform(channel=0)
    print("Finished querying waveform data.")

    # Save the waveform data to a CSV file
    save_to_csv(f"waveform_data_{command_name}_{file_number}.csv", data)

def save_to_csv(filename, data):
# Save data to CSV
    with open(filename, 'w', newline='') as file:
        import csv
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "Voltage (V)"])
        for x, y in zip(data['x'], data['y']):
            writer.writerow([x, y])

        print(f"Waveform data saved to file {filename}.")

with MSO5000(address=argv[1], rawMode=True) as mso:
    print(f"Identify: {mso.identify()}")

    client = ModbusTcpClient(MODBUS_SERVER_IP, port=MODBUS_SERVER_PORT)
    if not client.connect():
        print("Failed to connect to Modbus server!")
        exit()

    handler = ModbusClientHandler(client)

    mso.set_channel_enable(0, True)
    # mso.set_channel_enable(1, True)
    mso._scpi.scpiCommand(":TRIG:EDGE:SOUR CHAN1")
    # mso._scpi.scpiCommand(":TRIG:EDGE:SLOP POS")
    mso._scpi.scpiCommand(":TRIG:LEV 0.11")
    mso._scpi.scpiCommand(":SINGLE")  # Set to single acquisition mode
    mso._scpi.scpiCommand(":TRIG:MODE EDGE")  # Configure trigger (optional)

    mso._scpi.scpiCommand(":TRIG:POS 0.00001")  # Set trigger position to 10% of the screen
    mso._scpi.scpiCommand(":TIM:SCAL 0.000008")  # Adjust time base to 10 ms/div (example

    # Send a Modbus request
    #if response.isError():
     #   print("Error reading holding registers:", response)
    #else:
        # print("Holding Registers:", response.registers)
     #   pass

    count = 1
    for i in range(1, 100):

        commands = {"read_coils": {"address": random.choice(range(1, 255))}, 
                "write_coil": {"address": random.choice(range(1, 255)), "value": random.choice([True, False])},
                "read_holding_registers": {"address": random.choice(range(1, 255))},
                "write_register": {"address": random.choice(range(1,255)), "value": {random.randint(1, 2000)}, "slave": random.choice(range(1,255))},
                "read_input_registers": {"address": random.choice(range(1,255))}
                }

        for command, params in commands:
            execute_modbus_command(handler=handler, command_name=command, mso=mso, file_number=count, **params)
            count+=1