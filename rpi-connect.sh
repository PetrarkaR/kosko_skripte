#!/bin/bash

# Install RPi-Connect (Change to 'rpi-connect-lite' if needed)
sudo apt update && sudo apt full-upgrade -y
sudo apt install rpi-connect -y

# Enable user lingering
sudo loginctl enable-linger $USER

# Start Connect and enable auto-start
rpi-connect on

# Optional: Verify status
echo "Waiting 10 seconds for the service to start..."
sleep 10
systemctl --user status rpi-connect.service

# Final message
echo -e "\nReboot your Pi to test: sudo reboot"
