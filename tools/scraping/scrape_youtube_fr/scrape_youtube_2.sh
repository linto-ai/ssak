#!/bin/bash

FOLDER=YouTubeFr

while [ 1 -gt 0 ];do

    echo "Download mp4..."
    python3 scrape_youtube_download_audio.py $FOLDER || exit $?

    echo "Get metadata..."
    python3 scrape_youtube_get_metadata.py \
        $FOLDER/fr \
        -o $FOLDER/metadata \
        --ignore_if_exists \
        -c yt_scrape_check.tsv \
        -p yt_scrape_metadata.json || exit $?

    echo "Measure timing..."
    python3 ../total_duration.py YouTubeFr/mp3
    sleep 3600

done