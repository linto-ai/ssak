#!/bin/bash

while [ 1 -gt 0 ];do

    echo "Download mp4..."
    python3 scrape_youtube_download_audio.py || exit 1
    echo "Measure timing..."
    soxi YouTubeFr/mp3/* | tail -n 2
    sleep 3600

done