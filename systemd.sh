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
WRAPPER_SCRIPT="${SCRIPT_DIR}/run_kosko.sh"

# Check if the script exists
if [ ! -f "${SCRIPT_PATH}" ]; then
  echo "Error: Script ${SCRIPT_PATH} not found!"
  exit 1
fi

# Create wrapper script that does git pull before running
cat > "${WRAPPER_SCRIPT}" << 'EOF'
#!/bin/bash
# Wrapper script to update from git and run koskoPi.py

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Log start time
echo "[$(date)] Starting Kosko Script Service"

# Check if this is a git repository
if [ -d ".git" ]; then
  echo "[$(date)] Pulling latest changes from git..."
  
  # Stash any local changes (just in case)
  git stash
  
  # Pull latest changes
  if git pull origin main 2>&1; then
    echo "[$(date)] Git pull successful"
  else
    echo "[$(date)] Git pull failed, attempting with master branch..."
    git pull origin master 2>&1 || echo "[$(date)] Git pull failed on both branches"
  fi
  
  # Pop stashed changes if any
  git stash pop 2>/dev/null || true
else
  echo "[$(date)] Warning: Not a git repository, skipping update"
fi

# Run the Python script with provided arguments
echo "[$(date)] Starting koskoPi.py..."
exec /usr/bin/python "${SCRIPT_DIR}/koskoPi.py" "$@"
EOF

# Make wrapper script executable
chmod +x "${WRAPPER_SCRIPT}"
chown "${USER_NAME}:${USER_NAME}" "${WRAPPER_SCRIPT}"

# Create systemd service file
cat > "${SERVICE_PATH}" << EOF
[Unit]
Description=Kosko Script Service
After=network.target

[Service]
Type=simple
User=${USER_NAME}
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${WRAPPER_SCRIPT} jot.hef in.mp4 out.mp4
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
echo "Wrapper script created at: ${WRAPPER_SCRIPT}"
echo ""
echo "To check status: sudo systemctl status ${SERVICE_NAME}"
echo "To view logs: sudo journalctl -u ${SERVICE_NAME} -f"
echo "To stop service: sudo systemctl stop ${SERVICE_NAME}"
echo "To restart service: sudo systemctl restart ${SERVICE_NAME}"
echo "To disable service: sudo systemctl disable ${SERVICE_NAME}"

# Check immediate status
echo -e "\nCurrent service status:"
systemctl status "${SERVICE_NAME}"
