#!/bin/bash


echo "Setting up connection..."

ip addr add 10.10.10.30/24 dev enp3s0f3u3
ip addr add 10.10.11.30/24 dev enp3s0f3u3
ethtool -s enp3s0f3u3 speed 10 autoneg off duplex full 

echo "Setup completed!"
