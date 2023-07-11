#!/bin/bash

while [ 1 -gt 0 ];do

    echo "Download mp4..."
    python3 scrape_youtube_download_mp4.py || exit 1
    echo "Convert to mp3..."
    python3 convert_mp4_to_mp3.py YouTubeFr/mp4 YouTubeFr/mp3 || exit 1
    echo "Measure timing..."
    soxi YouTubeFr/mp3/* | tail -n 2
    sleep 3600

done