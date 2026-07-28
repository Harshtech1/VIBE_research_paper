#!/bin/bash
set -e

cd datasets/GECV/GECV-GroupVid

echo "Extracting negative..."
tar -xJf negative/negative.tar.xz -C negative

echo "Extracting neutral..."
tar -xJf neutral/neutral.tar.xz -C neutral

echo "Extracting positive..."
tar -xJf positive/happy.tar.xz -C positive

echo
echo "Extraction finished."

echo
echo "Video count:"
find . \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" \) | wc -l

echo
echo "First few videos:"
find . \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" \) | head -20
