#!/bin/bash

while [ 1 -gt 0 ];do
    
    FOLDER=YouTubeFr

    echo "Get metadata..."
    python3 scrape_youtube_get_metadata.py \
        $FOLDER/fr \
        -o $FOLDER/metadata \
        --ignore_if_exists \
        -c yt_scrape_check.tsv \
        -p yt_scrape_metadata.json || exit $?

    sleep 3600

done