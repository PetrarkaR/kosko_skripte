

echo "dtparam=pciex1_gen=3" | sudo tee -a /boot/firmware/config.txt

sudo apt install hailo-all -y
sudo reboot
