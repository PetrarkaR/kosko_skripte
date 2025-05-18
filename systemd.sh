#!/bin/bash
# Script to set up a systemd service for running koskoPi.py on Raspberry Pi 5
# Usage: sudo bash setup_kosko_service.sh

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script as root (use sudo)"
  exit 1
fi

# Define paths
SERVICE_NAME="kosko-script"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
USER_NAME="raspberry"
SCRIPT_DIR="/home/${USER_NAME}/kosko_skripte"
SCRIPT_PATH="${SCRIPT_DIR}/koskoPi.py"

# Check if the script exists
if [ ! -f "${SCRIPT_PATH}" ]; then
  echo "Error: Script ${SCRIPT_PATH} not found!"
  exit 1
fi

# Create systemd service file
cat > "${SERVICE_PATH}" << EOF
[Unit]
Description=Kosko Script Service
After=network.target

[Service]
Type=simple
User=${USER_NAME}
WorkingDirectory=${SCRIPT_DIR}
ExecStart=/usr/bin/python ${SCRIPT_PATH} jot.hef in.mp4 out.mp4
Restart=on-failure
RestartSec=5
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=kosko-script

[Install]
WantedBy=multi-user.target
EOF

# Set proper permissions
chmod 644 "${SERVICE_PATH}"

# Reload systemd to recognize the new service
systemctl daemon-reload

# Enable and start the service
systemctl enable "${SERVICE_NAME}.service"
systemctl start "${SERVICE_NAME}.service"

# Display status
echo "Service ${SERVICE_NAME} has been created, enabled, and started."
echo "To check status: sudo systemctl status ${SERVICE_NAME}"
echo "To view logs: sudo journalctl -u ${SERVICE_NAME}"
echo "To stop service: sudo systemctl stop ${SERVICE_NAME}"
echo "To disable service: sudo systemctl disable ${SERVICE_NAME}"

# Check immediate status
echo -e "\nCurrent service status:"
systemctl status "${SERVICE_NAME}"
