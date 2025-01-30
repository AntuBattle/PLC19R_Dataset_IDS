# ModBus analog dataset and IDS
This project aims at developing an IDS model that detects and blocks unauthorized devices from sending signals in a ModBus network.

The ModBus protocol does not account for any type of encription or privacy. 
This is because the protocol has been originally designed for industrial systems, where privacy and security were not thought of as a primary concern.

This IDS, together with other analyzers that block traffic at the data link layer, presents itself as a potential step forward towards securing ModBus communications.

## Directory Structure
The directories are organized as follows:
- `dataset-generation-scripts`: Where the scripts used to simulate a ModBus server and client are located (`server.py` and `client.py`).
Moreover, an `oscilloscope.py` script is present, which has allowed synchronization between signal emissions from the client and signal storing from the oscilloscope.

- `csv`: The main dataset is present in this folder, where every csv file represents a command, sampled with 10ns $DELTA$t.

- `model`: The main model scripts are stored in this folder. These scripts have been used to train an autoencoder to recognize authorized devices.


## The Testbed
The signals have been sampled with a Rigol MSO5104 Oscilloscope, and have been intercepted from an RJ45 Ethernet interface belonging to an Industrial Shields 19R PLC, powered by a Raspberry Pi 4 Model B.
This device has been assumed as an authorized device since it is a popular choice within the industry.

