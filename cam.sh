#!/bin/bash
sudo iwconfig wlan0 power off
# Ensure script is run as root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root. Use sudo."
    exit 1
fi

# Update and upgrade the system
apt update && apt full-upgrade -y

# Enable necessary overlays in config.txt
grep -qxF "dtoverlay=imx219" /boot/firmware/config.txt || echo "dtoverlay=imx219" >> /boot/firmware/config.txt

grep -qxF "camera_auto_detect=0" /boot/firmware/config.txt || echo "camera_auto_detect=0" >> /boot/firmware/config.txt

grep -qxF "start_x=1" /boot/firmware/config.txt || echo "start_x=1" >> /boot/firmware/config.txt

grep -qxF "gpu_mem=128" /boot/firmware/config.txt || echo "gpu_mem=128" >> /boot/firmware/config.txt


# Reboot to apply changes
echo "Setup complete. Rebooting now..."
