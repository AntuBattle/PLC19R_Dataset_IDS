from pymodbus.server import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
import logging

# Configure IP Address and Port to listen on
SERVER_IP_ADDRESS = "10.10.11.30"
SERVER_PORT = "50222"


# Configure logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# Create a Modbus data block
# This block simulates holding registers (40001-40010)
store = ModbusSlaveContext(
    di=ModbusSequentialDataBlock(0, [0] * 65536),  # Discrete Inputs
    co=ModbusSequentialDataBlock(0, [0] * 65536),  # Coils
    hr=ModbusSequentialDataBlock(0, [0] * 65536),  # Holding Registers
    ir=ModbusSequentialDataBlock(0, [0] * 65536)   # Input Registers
)
context = ModbusServerContext(slaves=store, single=True)

# Add device identification
identity = ModbusDeviceIdentification()
identity.VendorName = 'MyPLC'
identity.ProductCode = 'PLC1000'
identity.VendorUrl = 'https://example.com'
identity.ProductName = 'My PLC Modbus Server'
identity.ModelName = 'MyPLC v1.0'
identity.MajorMinorRevision = '1.0'

# Start the Modbus TCP server
if __name__ == "__main__":
    logging.info("Starting Modbus TCP Server...")
    StartTcpServer(context=context, identity=identity, address=(SERVER_IP_ADDRESS, SERVER_PORT))
