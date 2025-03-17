#!bin/bash

sudo chmod +x eth1.sh
sudo chmod +x cam.sh
sudo chmod +x rpi-connect.sh
sudo chmod +x hailo.sh

sudo ./eth1.sh -y
sudo ./cam.sh -y
sudo ./rpi-connect.sh -y
sudo ./hailo.sh -y
