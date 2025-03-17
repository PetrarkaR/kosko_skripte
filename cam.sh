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
grep -qxF "dtoverlay=imx219" /boot/firmware/config.txt || echo "dtoverlay=imx219,cam0" >> /boot/firmware/config.txt
echo "dtoverlay=imx219,cam1" >> /boot/firmware/config.txt
grep -qxF "camera_auto_detect=0" /boot/firmware/config.txt || echo "camera_auto_detect=0" >> /boot/firmware/config.txt
sed -i.bak '/camera_auto_detect=1/d' /boot/firmware/config.txt # will delete lines containing "NOT FOR RELEASE."

grep -qxF "start_x=1" /boot/firmware/config.txt || echo "start_x=1" >> /boot/firmware/config.txt

grep -qxF "gpu_mem=128" /boot/firmware/config.txt || echo "gpu_mem=128" >> /boot/firmware/config.txt

sudo wget https://www.waveshare.net/w/upload/7/7a/Imx290.zip
sudo unzip Imx290.zip
sudo cp imx290.json /usr/share/libcamera/ipa/rpi/pisp


# Reboot to apply changes
echo "Setup complete. Rebooting now..."
