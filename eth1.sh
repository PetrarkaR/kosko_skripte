#!/bin/bash

# 0. Safety checks
if [ $(id -u) -ne 0 ]; then
    echo "This script must be run as root" >&2
    exit 1
fi

# 1. System updates - REMOVED rpi-update as it often causes boot issues
echo "▶▶ Performing safe system updates..."
apt update && apt full-upgrade -y

# 2. Kernel module configuration with validation
echo "▶▶ Configuring kernel modules..."

# Backup original files
cp /boot/firmware/config.txt /boot/firmware/config.txt.bak
cp /boot/firmware/cmdline.txt /boot/firmware/cmdline.txt.bak
cp /etc/modules /etc/modules.bak

# Add dwc2 overlay safely
if ! grep -q "dtoverlay=dwc2" /boot/firmware/config.txt; then
    echo "dtoverlay=dwc2" >> /boot/firmware/config.txt
fi

# Add modules-load without duplicates
if ! grep -q "modules-load=dwc2" /boot/firmware/cmdline.txt; then
    sed -i 's/rootwait/rootwait modules-load=dwc2/g' /boot/firmware/cmdline.txt
fi

# Add libcomposite to modules
if ! grep -q "libcomposite" /etc/modules; then
    echo "libcomposite" >> /etc/modules
fi

# 3. USB gadget script with MAC address validation
echo "▶▶ Creating USB gadget configuration..."
cat > /usr/local/sbin/usb-gadget.sh <<'EOL'
#!/bin/bash

# Exit on error
set -e

validate_mac() {
    local mac=$1
    if ! [[ $mac =~ ^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$ ]]; then
        echo "Invalid MAC address: $mac" >&2
        exit 1
    fi
}

HOST_MAC="00:dc:c8:f7:75:15"
SELF_MAC="00:dd:dc:eb:6d:a1"
HOST_RNDIS="00:dc:c8:f7:75:16"
SELF_RNDIS="00:dd:dc:eb:6d:a2"

validate_mac $HOST_MAC
validate_mac $SELF_MAC
validate_mac $HOST_RNDIS
validate_mac $SELF_RNDIS

cd /sys/kernel/config/usb_gadget/ || exit 1

# Cleanup existing configuration
if [ -d display-pi ]; then
    echo "▶ Removing existing gadget..."
    rm -rf display-pi
fi

echo "▶ Creating new gadget..."
mkdir -p display-pi
cd display-pi

echo 0x1d6b > idVendor
echo 0x0104 > idProduct
echo 0x0103 > bcdDevice
echo 0x0320 > bcdUSB
echo 2 > bDeviceClass

mkdir -p strings/0x409
echo "fedcba9876543213" > strings/0x409/serialnumber
echo "Ben Hardill" > strings/0x409/manufacturer
echo "Display-Pi USB Device" > strings/0x409/product

# ECM Configuration
echo "▶ Configuring ECM..."
mkdir -p functions/ecm.usb0
echo $HOST_MAC > functions/ecm.usb0/host_addr
echo $SELF_MAC > functions/ecm.usb0/dev_addr

mkdir -p configs/c.1/strings/0x409
echo "CDC" > configs/c.1/strings/0x409/configuration
echo 250 > configs/c.1/MaxPower
ln -s functions/ecm.usb0 configs/c.1/

# RNDIS Configuration
echo "▶ Configuring RNDIS..."
mkdir -p configs/c.2
echo 0x80 > configs/c.2/bmAttributes
echo 0x250 > configs/c.2/MaxPower
mkdir -p configs/c.2/strings/0x409
echo "RNDIS" > configs/c.2/strings/0x409/configuration

echo "1" > os_desc/use
echo "0xcd" > os_desc/b_vendor_code
echo "MSFT100" > os_desc/qw_sign

mkdir -p functions/rndis.usb0
echo $HOST_RNDIS > functions/rndis.usb0/dev_addr
echo $SELF_RNDIS > functions/rndis.usb0/host_addr
echo "RNDIS" > functions/rndis.usb0/os_desc/interface.rndis/compatible_id
echo "5162001" > functions/rndis.usb0/os_desc/interface.rndis/sub_compatible_id

ln -s functions/rndis.usb0 configs/c.2
ln -s configs/c.2 os_desc

echo "▶ Activating gadget..."
ls /sys/class/udc | head -1 > UDC

echo "▶ Waiting for network interfaces..."
sleep 5

echo "▶ Bringing up network connections..."
nmcli connection up bridge-br0 || true
nmcli connection up bridge-slave-usb0 || true
nmcli connection up bridge-slave-usb1 || true

echo "▶ Restarting dnsmasq..."
systemctl restart dnsmasq || true

echo "✅ USB gadget configuration complete!"
EOL

# 4. Systemd service with proper dependencies
echo "▶▶ Creating systemd service..."
cat > /lib/systemd/system/usbgadget.service <<EOL
[Unit]
Description=USB Gadget Configuration
After=systemd-modules-load.service
After=network.target
Requires=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/usb-gadget.sh
ExecStop=/bin/rm -rf /sys/kernel/config/usb_gadget/display-pi

[Install]
WantedBy=multi-user.target
EOL

# 5. Network configuration with validation
echo "▶▶ Configuring networking..."
if ! nmcli con show bridge-br0 &> /dev/null; then
    nmcli con add type bridge ifname br0 con-name bridge-br0 ip4 10.55.0.1/24
fi

if ! nmcli con show bridge-slave-usb0 &> /dev/null; then
    nmcli con add type bridge-slave ifname usb0 master br0 con-name bridge-slave-usb0
fi

if ! nmcli con show bridge-slave-usb1 &> /dev/null; then
    nmcli con add type bridge-slave ifname usb1 master br0 con-name bridge-slave-usb1
fi

# 6. Dnsmasq configuration
echo "▶▶ Configuring dnsmasq..."
cat > /etc/dnsmasq.d/br0.conf <<EOL
interface=br0
dhcp-range=10.55.0.2,10.55.0.6,255.255.255.248,1h
dhcp-option=3
EOL

# 7. Final setup
echo "▶▶ Finalizing configuration..."
chmod +x /usr/local/sbin/usb-gadget.sh
systemctl daemon-reload
systemctl enable usbgadget.service
systemctl enable dnsmasq

echo "✅ Script completed successfully!"
echo "⚠️ Please reboot to apply all changes: sudo reboot"
