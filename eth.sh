#!/bin/bash

# 1. Update system and firmware
sudo apt update && sudo apt full-upgrade -y

# 2. Configure kernel modules
# Add dwc2 overlay to /boot/config.txt
echo "dtoverlay=dwc2" | sudo tee -a /boot/firmware/config.txt

# Add modules-load to /boot/cmdline.txt (ensure no newline)
sudo sed -i 's/rootwait/rootwait modules-load=dwc2/' /boot/firmware/cmdline.txt

# 3. Create USB gadget configuration script
sudo tee /usr/local/sbin/usb-gadget.sh > /dev/null <<'EOL'
#!/bin/bash
cd /sys/kernel/config/usb_gadget/
mkdir -p display-pi
cd display-pi

# Device descriptors
echo 0x1d6b > idVendor    # Linux Foundation
echo 0x0104 > idProduct   # Multifunction Composite Gadget
echo 0x0103 > bcdDevice   # v1.0.3
echo 0x0320 > bcdUSB      # USB2
echo 2 > bDeviceClass

# Strings
mkdir -p strings/0x409
echo "fedcba9876543213" > strings/0x409/serialnumber
echo "Ben Hardill" > strings/0x409/manufacturer
echo "Display-Pi USB Device" > strings/0x409/product

# ECM Configuration
mkdir -p configs/c.1/strings/0x409
echo "CDC" > configs/c.1/strings/0x409/configuration
echo 250 > configs/c.1/MaxPower
mkdir -p functions/ecm.usb0
echo "00:dc:c8:f7:75:15" > functions/ecm.usb0/host_addr  # Host MAC
echo "00:dd:dc:eb:6d:a1" > functions/ecm.usb0/dev_addr   # Pi MAC
ln -s functions/ecm.usb0 configs/c.1/

# RNDIS Configuration (for Windows compatibility)
mkdir -p configs/c.2/strings/0x409
echo "RNDIS" > configs/c.2/strings/0x409/configuration
echo 0x250 > configs/c.2/MaxPower
echo "1" > os_desc/use
echo "0xcd" > os_desc/b_vendor_code
echo "MSFT100" > os_desc/qw_sign
mkdir -p functions/rndis.usb0
echo "00:dc:c8:f7:75:16" > functions/rndis.usb0/dev_addr
echo "00:dd:dc:eb:6d:a2" > functions/rndis.usb0/host_addr
echo "RNDIS" > functions/rndis.usb0/os_desc/interface.rndis/compatible_id
echo "5162001" > functions/rndis.usb0/os_desc/interface.rndis/sub_compatible_id
ln -s functions/rndis.usb0 configs/c.2
ln -s configs/c.2 os_desc

# Activate gadget
ls /sys/class/udc > UDC
udevadm settle -t 5 || true
EOL

# 4. Make script executable and create systemd service
sudo chmod +x /usr/local/sbin/usb-gadget.sh

sudo tee /lib/systemd/system/usbgadget.service > /dev/null <<EOL
[Unit]
Description=USB Gadget Mode
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/usb-gadget.sh

[Install]
WantedBy=sysinit.target
EOL

# 5. Enable service and configure networking
sudo systemctl enable usbgadget.service

# NetworkManager bridge setup (for shared connection)
sudo nmcli con add type bridge ifname br0
sudo nmcli con add type bridge-slave ifname usb0 master br0
sudo nmcli con add type bridge-slave ifname usb1 master br0

# 6. Install and configure dnsmasq
sudo apt-get install -y dnsmasq
sudo tee /etc/dnsmasq.d/br0 > /dev/null <<EOL
dhcp-authoritative
dhcp-rapid-commit
interface=br0
dhcp-range=10.55.0.2,10.55.0.6,255.255.255.248,1h
dhcp-option=3
EOL

# 7. Final reboot
echo "done"
