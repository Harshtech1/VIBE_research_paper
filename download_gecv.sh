#!/bin/bash

set -e

pip install -q gdown

mkdir -p datasets/GECV
cd datasets/GECV

echo "Downloading GECV-GroupVid..."

gdown --folder "https://drive.google.com/drive/folders/1scdDzqq1lqQSahKF7g1143jWSUQyPk23"

echo
echo "Download Complete!"
echo

find .
